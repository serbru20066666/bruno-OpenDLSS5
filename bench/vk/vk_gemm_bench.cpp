// Standalone Vulkan benchmark: packed-FP16 GEMM (gemm_f16_packed.comp) on the GEMM shapes of the NR network.
// Reports achieved TFLOPS per shape on any Vulkan 1.3 GPU (written for RDNA2, which has no matrix units).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "volk.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s failed: %d\n", #x, r_); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice phys;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  printf("no memory type\n");
  exit(1);
}

struct Buf { VkBuffer b; VkDeviceMemory m; };

static Buf makeBuf(VkDeviceSize size) {
  Buf r;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  CK(vkCreateBuffer(dev, &bi, nullptr, &r.b));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, r.b, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  CK(vkAllocateMemory(dev, &ai, nullptr, &r.m));
  CK(vkBindBufferMemory(dev, r.b, r.m, 0));
  return r;
}

int main(int argc, char** argv) {
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
  phys = VK_NULL_HANDLE;
  for (auto d : pds) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    printf("device: %s\n", p.deviceName);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && phys == VK_NULL_HANDLE) phys = d;
  }
  if (!phys) phys = pds[0];
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(phys, &props);
  printf("using: %s\n", props.deviceName);

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
  VkQueue q;
  vkGetDeviceQueue(dev, qi.queueFamilyIndex, 0, &q);

  std::ifstream f(argc > 1 ? argv[1] : "gemm_f16_packed.spv", std::ios::binary);
  std::vector<char> spv((std::istreambuf_iterator<char>(f)), {});
  if (spv.empty()) { printf("cannot read SPIR-V\n"); return 1; }
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spv.size();
  smi.pCode = (const uint32_t*)spv.data();
  VkShaderModule sm;
  CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));

  VkDescriptorSetLayoutBinding bd[3];
  for (int i = 0; i < 3; i++) bd[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dli.bindingCount = 3;
  dli.pBindings = bd;
  VkDescriptorSetLayout dl;
  CK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dl));
  VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &dl;
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pr;
  VkPipelineLayout pl;
  CK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.layout = pl;
  cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
  VkPipeline pipe;
  CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 16};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 16;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &ps;
  VkDescriptorPool dp;
  CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.queueFamilyIndex = qi.queueFamilyIndex;
  VkCommandPool cp;
  CK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpi.queryCount = 2;
  VkQueryPool qpool;
  CK(vkCreateQueryPool(dev, &qpi, nullptr, &qpool));

  struct Shape { uint32_t M, K, N; const char* what; };
  const std::vector<Shape> shapes = {
      {245760, 32, 128, "L0 FFN up (32ch)"},        {61440, 64, 128, "L1 expert W1 (64ch)"},
      {61440, 64, 64, "L1 W3/proj (64ch)"},         {15360, 128, 128, "L2 expert W1/W3 (128ch)"},
      {15360, 128, 384, "L2 QKV (128ch)"},          {3840, 256, 256, "L3 W3/proj (256ch)"},
      {3840, 256, 768, "L3 QKV (256ch)"},           {960, 512, 512, "L4 W3 (512ch)"},
      {960, 512, 1536, "L4 QKV (512ch)"},           {256, 1024, 4096, "ViT FFN up"},
      {256, 4096, 1024, "ViT FFN down"}};
  const int iters = 20;
  printf("note: M and N are padded up to multiples of 128; TFLOPS counts the padded work.\n");
  for (const Shape& s : shapes) {
    uint32_t M = (s.M + 127) / 128 * 128, N = (s.N + 127) / 128 * 128, K = s.K;
    Buf A = makeBuf((VkDeviceSize)M * K * 2), B = makeBuf((VkDeviceSize)K * N * 2), C = makeBuf((VkDeviceSize)M * N * 2);
    VkDescriptorSetLayout layouts[1] = {dl};
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = dp;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = layouts;
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dsa, &ds));
    VkDescriptorBufferInfo bi[3] = {{A.b, 0, VK_WHOLE_SIZE}, {B.b, 0, VK_WHOLE_SIZE}, {C.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++)
      w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr};
    vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cp;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bgi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bgi));
    vkCmdFillBuffer(cb, A.b, 0, VK_WHOLE_SIZE, 0x2E662E66);   // f16 ~0.1
    vkCmdFillBuffer(cb, B.b, 0, VK_WHOLE_SIZE, 0x2E662E66);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[3] = {M, N, K};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
    vkCmdResetQueryPool(cb, qpool, 0, 2);
    VkMemoryBarrier cm{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    auto dispatch = [&] {
      vkCmdDispatch(cb, N / 128, M / 128, 1);
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &cm, 0, nullptr, 0, nullptr);
    };
    for (int i = 0; i < 3; i++) dispatch();   // warm-up
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 0);
    for (int i = 0; i < iters; i++) dispatch();
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(q));
    uint64_t ts[2];
    CK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    double ms = (ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6 / iters;
    double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("%-26s M=%-7u K=%-5u N=%-5u  %8.3f ms  %6.2f TFLOPS\n", s.what, M, K, N, ms, tf);
    vkFreeCommandBuffers(dev, cp, 1, &cb);
    for (Buf* b : {&A, &B, &C}) { vkDestroyBuffer(dev, b->b, nullptr); vkFreeMemory(dev, b->m, nullptr); }
  }
  return 0;
}
