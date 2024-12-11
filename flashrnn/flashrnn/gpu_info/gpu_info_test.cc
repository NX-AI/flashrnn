#include <functional>
#include <pybind11/pybind11.h>
#include <vector>

#define DECLARE_AND_REGISTER(type, name)                                       \
  type name;                                                                   \
  static type get_##name() const { return name; }                              \
  static void set_##name(const type &value) { name = value; }                  \
  struct _register_##name {                                                    \
    _register_##name() {                                                       \
      binders.push_back([](pybind11::class_<SelfType> &cls) {                  \
        cls.def_property(#name, &SelfType::get_##name, &SelfType::set_##name); \
      });                                                                      \
    }                                                                          \
  } _instance_register_##name;

#define WRAP_MEMBERS_HELPER(x, type, name, ...)                                \
  DECLARE_AND_REGISTER(type, name)                                             \
  WRAP_MEMBERS_HELPER(x, __VA_ARGS__)

#define WRAP_MEMBERS_HELPER_END(x, ...)

#define WRAP_MEMBERS(...)                                                      \
  using SelfType = std::remove_reference<decltype(*this)>::type;               \
  static std::vector<std::function<void(pybind11::class_<SelfType> &)>>        \
      binders;                                                                 \
  WRAP_MEMBERS_HELPER(0, __VA_ARGS__, _, _, _)

struct A {
  WRAP_MEMBERS(int, a)
};

PYBIND11_MODULE(example, m) {
  pybind11::class_<A> cls(m, "A");
  for (auto &binder : A::binders) {
    binder(cls);
  }
}
