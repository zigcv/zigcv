#include "core.h"
#include "zig_core.h"

struct Mats Mats_New(int length){
    struct Mats mats;
    mats.length = length;
    mats.mats = new Mat[length];
    return mats;
}