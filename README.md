# ZIGCV

## Caution
Still under development, so the zig APIs will be dynamically changed.

Tested on zig version:0.10.0-dev.3024+47c58cba5

You can use `const c_api = @import("zigcv").c_api;` to call c bindings directly.  
This C-API is currently fixed.

## How to execute

At first, install openCV. (maybe you can read how to install from [here](https://github.com/hybridgroup/gocv#how-to-install)).  
Then:

```sh
git clone https://github.com/ryoppippi/zigcv
zig build 
```
## Demos 
you can build some demos.  
For example:
```sh
zig build examples
./zig-out/bin/face_detection 0
```
<img width="400" alt="face detection" src="https://user-images.githubusercontent.com/1560508/185785932-404865df-d2d1-4f6a-b3ec-18632f77f7ae.png">

You can see the full demo list by `zig build --help`.

## Technical restrictions

Due to zig being a relatively new language it does [not have full C ABI support](https://github.com/ziglang/zig/issues/1481) at the moment.  
For use that mainly means we can't use any functions that return structs that are less than 16 bytes large on x86, and passing structs to any functions may cause memory error on arm.

## License

MIT

## Author

Ryotaro "Justin" Kimura (a.k.a. ryoppippi)



