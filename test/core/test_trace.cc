#define BACKWARD_HAS_DW 1
#include "core/backward.hpp"
#include "test.h"

namespace infini {

int func_a( ) {
  int a = 123;
  int b = 0;
  int c = a / b;
  return c;
}

int func_b() {
  return func_a();
}

int func_c() {
  return func_b();
}

TEST(Trace, EasyTrace) {
  int res = func_c();
  EXPECT_NE(res, 5);
}

} // namespace infini
