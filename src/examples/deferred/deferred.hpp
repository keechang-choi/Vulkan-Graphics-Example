#include "Vge_base.hpp"

//
#include "vgeu_gltf.hpp"
#include "vgeu_texture.hpp"

// std
#include <vector>

namespace vge {

struct Options {
  int32_t debugDisplayarget = 0;
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  virtual void initVulkan();
  virtual void getEnabledExtensions();
  virtual void getEnabledFeatures();
  virtual void render();
  virtual void prepare();
  virtual void viewChanged();
  virtual void setupCommandLineParser(CLI::App& app);
  virtual void onUpdateUIOverlay();

  Options opts{};
};
}  // namespace vge