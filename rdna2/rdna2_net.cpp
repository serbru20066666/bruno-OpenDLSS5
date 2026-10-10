// RDNA2 backend, step 1: the whole NR network graph (71 blocks) in plain Vulkan compute with f16 packed math and
// SYNTHETIC weights (constant values). It measures real time per frame on GPUs without matrix units; the output is not
// a meaningful image. Graph and schedule follow docs/network.md and src/nr_graph.cpp (FP8 quantisation, the exact
// exp bit trick and the bit-exact reduction orders are not reproduced yet).
//
//   rdna2_net [width height]        default 1280 720; run from the directory holding the .spv files
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "volk.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static VkPhysicalDevice phys;
static VkDevice dev;
static VkPhysicalDeviceProperties props;
static std::vector<std::pair<VkBuffer, VkDeviceMemory>> allBuffers;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  printf("no memory type\n");
  exit(1);
}

struct View {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0, size = 0;
};

static View makeBuffer(VkDeviceSize size, VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkBuffer b;
  CK(vkCreateBuffer(dev, &bi, nullptr, &b));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, b, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = memType(mr.memoryTypeBits, flags);
  VkDeviceMemory m;
  CK(vkAllocateMemory(dev, &ai, nullptr, &m));
  CK(vkBindBufferMemory(dev, b, m, 0));
  allBuffers.push_back({b, m});
  return {b, 0, size};
}

static uint32_t pad128(uint32_t v) { return (v + 127) / 128 * 128; }
static uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

struct Act {
  View v;
  uint32_t tokens = 0, C = 0;
};

enum Shader { GEMM, WATTN, GATTN, OPS, SHADER_COUNT };
static const char* shaderFiles[SHADER_COUNT] = {"gemm.spv", "wattn.spv", "gattn.spv", "ops.spv"};

struct Net {
  VkQueue queue;
  uint32_t family;
  VkShaderModule modules[SHADER_COUNT];
  VkDescriptorSetLayout dsl;
  VkPipelineLayout pl;
  VkDescriptorPool dpool;
  VkCommandBuffer cb;
  VkQueryPool qpool;
  std::map<uint64_t, VkPipeline> pipes;
  std::vector<std::string> stageNames;
  uint32_t queries = 0, dispatches = 0;
  View weights, dummy;
  VkDeviceSize weightCursor = 0;
  View scratch[5];  // 0 hidden, 1 qkv, 2 mid, 3 attention, 4 y
  uint32_t pushBytes = 0;
  uint64_t weightHalves = 0;

  View W(uint64_t halves) {
    VkDeviceSize bytes = (halves * 2 + 255) / 256 * 256;
    if (weightCursor + bytes > weights.size) { printf("weight pool exhausted\n"); exit(1); }
    View v{weights.buffer, weightCursor, halves * 2};
    weightCursor += bytes;
    weightHalves += halves;
    return v;
  }
  View S(int slot, uint64_t bytes) {
    if (scratch[slot].size < bytes) scratch[slot] = makeBuffer(bytes);
    return scratch[slot];
  }
  Act newAct(uint32_t tokens, uint32_t C) { return {makeBuffer((VkDeviceSize)pad128(tokens) * C * 2), tokens, C}; }

  VkPipeline pipe(Shader s, uint32_t tn, uint32_t flags) {
    uint64_t key = (uint64_t)s << 40 | (uint64_t)tn << 8 | flags;
    auto it = pipes.find(key);
    if (it != pipes.end()) return it->second;
    uint32_t data[2] = {tn, flags};
    VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
    VkSpecializationInfo si{2, me, 8, data};
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.layout = pl;
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, modules[s], "main",
                s == GEMM ? &si : nullptr};
    VkPipeline p;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
    pipes[key] = p;
    return p;
  }

  void barrier() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  }

  void dispatch(Shader s, uint32_t tn, uint32_t flags, const View* bind[5], const void* push, uint32_t bytes, uint32_t x, uint32_t y,
                uint32_t z) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &ai, &ds));
    VkDescriptorBufferInfo bi[5];
    VkWriteDescriptorSet w[5];
    for (int i = 0; i < 5; i++) {
      const View& v = bind[i] ? *bind[i] : dummy;
      bi[i] = {v.buffer, v.offset, v.size};
      w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr};
    }
    vkUpdateDescriptorSets(dev, 5, w, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe(s, tn, flags));
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
    vkCmdDispatch(cb, x, y, z);
    barrier();
    dispatches++;
  }

  // C[M][N] = A * W (+silu) (+ R * scale), `batches` grouped GEMMs (see gemm.comp).
  void gemm(const View& A, const View& Wt, const View& C, const View* R, const View* Sc, uint32_t M, uint32_t N, uint32_t K, uint32_t lda,
            uint32_t ldc, uint32_t ldr, bool silu, uint32_t batches = 1, uint32_t aStride = 0, uint32_t cStride = 0, uint32_t rStride = 0) {
    uint32_t tn = N % 128 == 0 ? 128 : N % 64 == 0 ? 64 : 32;
    if (N % 32) { printf("N=%u not a multiple of 32\n", N); exit(1); }
    uint32_t push[10] = {M, N, K, lda, ldc, ldr, aStride, cStride, rStride, K * N / 2};
    const View* bind[5] = {&A, &Wt, &C, R, Sc};
    dispatch(GEMM, tn, (silu ? 1u : 0u) | (R ? 2u : 0u), bind, push, sizeof push, N / tn, (M + 127) / 128, batches);
  }

  void pool(const Act& in, Act& out, uint32_t w, uint32_t h) {
    uint32_t push[4] = {0, w, h, in.C};
    const View* bind[5] = {&in.v, nullptr, &out.v, nullptr, nullptr};
    dispatch(OPS, 0, 0, bind, push, sizeof push, (w * h * (in.C / 2) + 255) / 256, 1, 1);
  }
  void upsampleAdd(const Act& low, const Act& skip, const View& scale, Act& out, uint32_t w, uint32_t h) {
    uint32_t push[4] = {1, w, h, out.C};
    const View* bind[5] = {&low.v, nullptr, &out.v, &skip.v, &scale};
    dispatch(OPS, 0, 0, bind, push, sizeof push, (w * h * (out.C / 2) + 255) / 256, 1, 1);
  }

  void mark(const char* name) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, queries++);
    stageNames.push_back(name);
  }

  // One window block (C = 32 .. 512): FFN + skip, QKV, shifted-window attention, projection + skip.
  uint32_t phase[8] = {};
  Act windowBlock(const Act& x, uint32_t w, uint32_t h, uint32_t C, int level) {
    static const uint32_t shifts[4][2] = {{0, 0}, {4, 4}, {4, 0}, {0, 4}};
    uint32_t sx = shifts[phase[level] & 3][0], sy = shifts[phase[level] & 3][1];
    phase[level]++;
    uint32_t T = w * h, Tp = pad128(T);
    View sc = W(C);
    View hid, mid = S(2, (uint64_t)Tp * C * 2), y = S(4, (uint64_t)Tp * C * 2);
    if (C == 32) {
      hid = S(0, (uint64_t)Tp * 128 * 2);
      gemm(x.v, W(32 * 128), hid, nullptr, nullptr, T, 128, 32, 32, 128, 0, true);
      gemm(hid, W(128 * 32), y, &x.v, &sc, T, 32, 128, 128, 32, 32, false);
    } else if (C <= 256) {
      uint32_t E = C / 32;
      hid = S(0, (uint64_t)Tp * E * 128 * 2);
      gemm(x.v, W((uint64_t)E * C * 128), hid, nullptr, nullptr, T, 128, C, C, E * 128, 0, true, E, 0, 128);
      gemm(hid, W((uint64_t)E * 128 * 32), mid, nullptr, nullptr, T, 32, 128, E * 128, C, 0, false, E, 128, 32);
      gemm(mid, W((uint64_t)C * C), y, &x.v, &sc, T, C, C, C, C, C, C, false);
    } else {  // 512: eight 64 -> 256 -> 64 branches, then 512 -> 512 carrying the skip
      hid = S(0, (uint64_t)Tp * 2048 * 2);
      gemm(x.v, W(8 * 64 * 256), hid, nullptr, nullptr, T, 256, 64, 512, 2048, 0, true, 8, 64, 256);
      gemm(hid, W(8 * 256 * 64), mid, nullptr, nullptr, T, 64, 256, 2048, 512, 0, false, 8, 256, 64);
      gemm(mid, W(512 * 512), y, &x.v, &sc, T, 512, 512, 512, 512, 512, false);
    }
    View qkv = S(1, (uint64_t)Tp * 3 * C * 2), att = S(3, (uint64_t)Tp * C * 2);
    gemm(y, W((uint64_t)C * 3 * C), qkv, nullptr, nullptr, T, 3 * C, C, C, 3 * C, 0, false);
    uint32_t nwx = (w + sx + 7) / 8, nwy = (h + sy + 7) / 8, heads = C / 32;
    float qscale = 1.0f;
    uint32_t push[8] = {w, h, C, heads, sx, sy, nwx, 0};
    memcpy(&push[7], &qscale, 4);
    View prior = W((uint64_t)heads * 4096);
    const View* ab[5] = {&qkv, &prior, &att, nullptr, nullptr};
    dispatch(WATTN, 0, 0, ab, push, sizeof push, nwx * nwy, heads, 1);
    Act out = newAct(T, C);
    View sc2 = W(C);
    gemm(att, W((uint64_t)C * C), out.v, &y, &sc2, T, C, C, C, C, C, false);
    return out;
  }

  Act vitBlock(const Act& x) {
    const uint32_t C = 1024, T = x.tokens, Tp = pad128(T);
    View sc = W(C), sc2 = W(C);
    View hid = S(0, (uint64_t)Tp * 4096 * 2), y = S(4, (uint64_t)Tp * C * 2);
    gemm(x.v, W((uint64_t)C * 4096), hid, nullptr, nullptr, T, 4096, C, C, 4096, 0, true);
    gemm(hid, W((uint64_t)4096 * C), y, &x.v, &sc, T, C, 4096, 4096, C, C, false);
    View qkv = S(1, (uint64_t)Tp * 3 * C * 2), att = S(3, (uint64_t)Tp * C * 2);
    gemm(y, W((uint64_t)C * 3 * C), qkv, nullptr, nullptr, T, 3 * C, C, C, 3 * C, 0, false);
    uint32_t push[4] = {T, C, 32, 0};
    float qs = 5.65685f;  // sqrt(32)
    memcpy(&push[3], &qs, 4);
    const View* ab[5] = {&qkv, nullptr, &att, nullptr, nullptr};
    dispatch(GATTN, 0, 0, ab, push, sizeof push, (T * 32 + 63) / 64, 1, 1);
    Act out = newAct(T, C);
    gemm(att, W((uint64_t)C * C), out.v, &y, &sc2, T, C, C, C, C, C, false);
    return out;
  }

  Act encoderTransition(const Act& x, uint32_t w, uint32_t h, uint32_t cout) {
    Act pooled = newAct(w * h, x.C);
    pool(x, pooled, w, h);
    if (cout == x.C) return pooled;
    Act out = newAct(w * h, cout);
    gemm(pooled.v, W((uint64_t)x.C * cout), out.v, nullptr, nullptr, w * h, cout, x.C, x.C, cout, 0, false);
    return out;
  }
  Act decoderTransition(const Act& low, const Act& skip, uint32_t lw, uint32_t lh, uint32_t cout) {
    Act proj = newAct(lw * lh, cout);
    gemm(low.v, W((uint64_t)low.C * cout), proj.v, nullptr, nullptr, lw * lh, cout, low.C, low.C, cout, 0, false);
    Act out = newAct(lw * 2 * lh * 2, cout);
    upsampleAdd(proj, skip, W(cout), out, lw * 2, lh * 2);
    return out;
  }
};

struct Dim { uint32_t w, h; };

int main(int argc, char** argv) {
  uint32_t W = argc > 2 ? atoi(argv[1]) : 1280, H = argc > 2 ? atoi(argv[2]) : 720;
  CK(volkInitialize());
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  CK(vkCreateInstance(&ici, nullptr, &inst));
  volkLoadInstance(inst);
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(inst, &n, pds.data());
  for (auto d : pds) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && !phys) phys = d;
  }
  if (!phys) phys = pds[0];
  vkGetPhysicalDeviceProperties(phys, &props);
  printf("GPU: %s | %ux%u\n", props.deviceName, W, H);

  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.shaderFloat16 = VK_TRUE;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  f11.storageBuffer16BitAccess = VK_TRUE;
  f11.pNext = &f12;
  float prio = 1;
  VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qi.queueCount = 1;
  qi.pQueuePriorities = &prio;
  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qp(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qp.data());
  for (uint32_t i = 0; i < qn; i++)
    if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi.queueFamilyIndex = i; break; }
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f11;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qi;
  CK(vkCreateDevice(phys, &dci, nullptr, &dev));
  volkLoadDevice(dev);

  Net net;
  net.family = qi.queueFamilyIndex;
  vkGetDeviceQueue(dev, net.family, 0, &net.queue);
  for (int i = 0; i < SHADER_COUNT; i++) {
    std::ifstream f(shaderFiles[i], std::ios::binary);
    std::vector<char> spv((std::istreambuf_iterator<char>(f)), {});
    if (spv.empty()) { printf("cannot read %s (run from the directory with the .spv files)\n", shaderFiles[i]); return 1; }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spv.size();
    smi.pCode = (const uint32_t*)spv.data();
    CK(vkCreateShaderModule(dev, &smi, nullptr, &net.modules[i]));
  }
  VkDescriptorSetLayoutBinding bd[5];
  for (int i = 0; i < 5; i++) bd[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dli.bindingCount = 5;
  dli.pBindings = bd;
  CK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &net.dsl));
  VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &net.dsl;
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pr;
  CK(vkCreatePipelineLayout(dev, &pli, nullptr, &net.pl));
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * 4096};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 4096;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &ps;
  CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &net.dpool));
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.queueFamilyIndex = net.family;
  VkCommandPool cpool;
  CK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = cpool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  CK(vkAllocateCommandBuffers(dev, &cai, &net.cb));
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpi.queryCount = 128;
  CK(vkCreateQueryPool(dev, &qpi, nullptr, &net.qpool));

  net.weights = makeBuffer(512ull << 20);
  net.dummy = makeBuffer(4096);
  View readback = makeBuffer(256, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkDeviceMemory readbackMemory = allBuffers.back().second;

  // Geometry (approximation of Geometry::fromValid): padded field, then six halvings aligned to 4.
  uint32_t fw = alignUp(W, 64), fh = alignUp(H, 64);
  std::vector<Dim> L;
  Dim d{fw, fh};
  for (int i = 0; i < 6; i++) {
    d = {alignUp((d.w + 1) / 2, 4), alignUp((d.h + 1) / 2, 4)};
    L.push_back(d);
  }
  printf("field %ux%u, levels:", fw, fh);
  for (auto& l : L) printf(" %ux%u", l.w, l.h);
  printf("\n");

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CK(vkBeginCommandBuffer(net.cb, &beginInfo));
  vkCmdResetQueryPool(net.cb, net.qpool, 0, 128);
  vkCmdFillBuffer(net.cb, net.weights.buffer, 0, VK_WHOLE_SIZE, 0x18081808);  // f16 ~0.002
  vkCmdFillBuffer(net.cb, net.dummy.buffer, 0, VK_WHOLE_SIZE, 0);
  Act feat = net.newAct(fw * fh, 16);
  vkCmdFillBuffer(net.cb, feat.v.buffer, 0, VK_WHOLE_SIZE, 0x30003000);  // f16 0.125 input features
  net.barrier();
  net.mark("start");

  // Encoder
  Act b0in = net.newAct(fw * fh, 32);
  net.gemm(feat.v, net.W(16 * 32), b0in.v, nullptr, nullptr, fw * fh, 32, 16, 16, 32, 0, false);
  Act b0 = net.windowBlock(b0in, fw, fh, 32, 6);
  net.mark("block 0 (full res, 32ch)");
  Act x = net.encoderTransition(b0, L[0].w, L[0].h, 32);
  for (int i = 0; i < 4; i++) x = net.windowBlock(x, L[0].w, L[0].h, 32, 0);
  Act skip32 = x;
  net.mark("enc L0 32ch x4");
  const struct { uint32_t C; int blocks; int level; const char* name; } enc[] = {{64, 4, 1, "enc L1 64ch x4"}, {128, 6, 2, "enc L2 128ch x6"},
                                                                                {256, 8, 3, "enc L3 256ch x8"}, {512, 8, 4, "enc L4 512ch x8"}};
  Act skips[5];
  skips[0] = skip32;
  for (int s = 0; s < 4; s++) {
    x = net.encoderTransition(x, L[enc[s].level].w, L[enc[s].level].h, enc[s].C);
    for (int i = 0; i < enc[s].blocks; i++) x = net.windowBlock(x, L[enc[s].level].w, L[enc[s].level].h, enc[s].C, enc[s].level);
    skips[s + 1] = x;
    net.mark(enc[s].name);
  }
  // ViT
  x = net.encoderTransition(x, L[5].w, L[5].h, 1024);
  for (int i = 0; i < 8; i++) x = net.vitBlock(x);
  net.mark("ViT 1024ch x8 (global attention)");
  // Decoder
  const struct { uint32_t C; int blocks; int level; const char* name; } dec[] = {{512, 8, 4, "dec L4 512ch x8"}, {256, 8, 3, "dec L3 256ch x8"},
                                                                                {128, 6, 2, "dec L2 128ch x6"}, {64, 4, 1, "dec L1 64ch x4"},
                                                                                {32, 4, 0, "dec L0 32ch x4"}};
  for (int s = 0; s < 5; s++) {
    int lv = dec[s].level;
    if (L[lv + 1].w * 2 != L[lv].w || L[lv + 1].h * 2 != L[lv].h) {
      printf("level %d is not exactly half of level %d at %ux%u; pick another size (e.g. 1280x720, 1920x1080)\n", lv + 1, lv, W, H);
      return 1;
    }
    x = net.decoderTransition(x, skips[4 - s], L[lv + 1].w, L[lv + 1].h, dec[s].C);
    for (int i = 0; i < dec[s].blocks; i++) x = net.windowBlock(x, L[lv].w, L[lv].h, dec[s].C, lv);
    net.mark(dec[s].name);
  }
  {  // 69 -> 70: upsample to the field, add block 0 output, block 70, head
    Act up = net.newAct(fw * fh, 32);
    net.upsampleAdd(x, b0, net.W(32), up, fw, fh);
    Act b70 = net.windowBlock(up, fw, fh, 32, 6);
    Act head = net.newAct(fw * fh, 32);  // 32 -> 4 head, padded to 32 columns
    net.gemm(b70.v, net.W(32 * 32), head.v, nullptr, nullptr, fw * fh, 32, 32, 32, 32, 0, false);
    net.mark("block 70 + head (full res)");
    VkBufferCopy bc{0, 0, 64};
    vkCmdCopyBuffer(net.cb, head.v.buffer, readback.buffer, 1, &bc);
  }
  CK(vkEndCommandBuffer(net.cb));
  printf("recorded %u dispatches, synthetic weights %.0f MiB (f16)\n", net.dispatches, net.weightHalves * 2 / 1048576.0);

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &net.cb;
  double best = 1e9, sum = 0;
  std::vector<uint64_t> ts(net.queries), bestTs;
  const int runs = 8;
  for (int r = 0; r < runs; r++) {
    CK(vkQueueSubmit(net.queue, 1, &si, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(net.queue));
    CK(vkGetQueryPoolResults(dev, net.qpool, 0, net.queries, ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    double ms = (ts.back() - ts.front()) * props.limits.timestampPeriod * 1e-6;
    printf("run %d: %.1f ms\n", r, ms);
    if (r > 0) { sum += ms; if (ms < best) { best = ms; bestTs = ts; } }
  }
  printf("\nstage breakdown (best run):\n");
  for (uint32_t i = 1; i < net.queries; i++)
    printf("  %-34s %8.2f ms\n", net.stageNames[i].c_str(), (bestTs[i] - bestTs[i - 1]) * props.limits.timestampPeriod * 1e-6);
  printf("\nTOTAL best %.1f ms (%.1f fps) | mean %.1f ms | %ux%u on %s\n", best, 1000 / best, sum / (runs - 1), W, H, props.deviceName);
  void* mapped;
  vkMapMemory(dev, readbackMemory, 0, 64, 0, &mapped);
  const uint16_t* hv = (const uint16_t*)mapped;
  printf("head[0..3] raw f16: %04x %04x %04x %04x (synthetic weights: values are not an image)\n", hv[0], hv[1], hv[2], hv[3]);
  return 0;
}
