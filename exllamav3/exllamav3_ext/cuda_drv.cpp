#include <cstdio>
#include <c10/util/Exception.h>
#include "cuda_drv.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DRV_STR2(x) #x
#define DRV_STR(x) DRV_STR2(x)

static void* drv_sym(void* lib, const char* name)
{
    #ifdef _WIN32
        void* fp = (void*) GetProcAddress((HMODULE) lib, name);
    #else
        void* fp = dlsym(lib, name);
    #endif
    TORCH_CHECK(fp, "CUDA driver symbol not found: ", name);
    return fp;
}

const CudaDrv& CudaDrv::instance()
{
    static CudaDrv d = []
    {
        #ifdef _WIN32
            void* lib = (void*) LoadLibraryA("nvcuda.dll");
        #else
            #ifdef __HIP_PLATFORM_AMD__
            // Prefer the HIP runtime this process already runs on (torch / the extension link it by
            // soname); a bare "libamdhip64.so" can resolve to a different ROCm install (e.g. /opt/rocm
            // under a pip ROCm SDK), and launching through a second runtime crashes
            void* lib = nullptr;
            for (const char* name : { "libamdhip64.so.7", "libamdhip64.so.8", "libamdhip64.so.9", "libamdhip64.so.10", "libamdhip64.so.6" })
                if ((lib = dlopen(name, RTLD_NOW | RTLD_NOLOAD))) break;
            if (!lib) lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
#else
            void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
#endif
            #ifndef __HIP_PLATFORM_AMD__
        if (!lib) lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
#endif
        #endif
        TORCH_CHECK(lib, "Could not load the CUDA driver library");

        CudaDrv d{};
        d.module_load_data                  = (decltype(&cuModuleLoadData))               drv_sym(lib, DRV_STR(cuModuleLoadData));
        d.module_unload                     = (decltype(&cuModuleUnload))                 drv_sym(lib, DRV_STR(cuModuleUnload));
        d.module_get_function               = (decltype(&cuModuleGetFunction))            drv_sym(lib, DRV_STR(cuModuleGetFunction));
        d.func_set_attribute                = (decltype(&cuFuncSetAttribute))             drv_sym(lib, DRV_STR(cuFuncSetAttribute));
        d.launch_kernel                     = (decltype(&cuLaunchKernel))                 drv_sym(lib, DRV_STR(cuLaunchKernel));
        d.graph_kernel_node_get_params      = (decltype(&cuGraphKernelNodeGetParams))     drv_sym(lib, DRV_STR(cuGraphKernelNodeGetParams));
        d.graph_exec_kernel_node_set_params = (decltype(&cuGraphExecKernelNodeSetParams)) drv_sym(lib, DRV_STR(cuGraphExecKernelNodeSetParams));
        return d;
    }
    ();
    return d;
}
