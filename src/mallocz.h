#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void *malloc(size_t size);
extern void *realloc(void *ptr, size_t size);
extern void free(void *ptr);

#ifdef __cplusplus
}

#endif

#ifdef __cplusplus
void *operator new(size_t size) { return malloc(size); }
void *operator new[](size_t size) { return malloc(size); }

void operator delete(void *p) noexcept { free(p); }
void operator delete[](void *p) noexcept { free(p); }
#endif
