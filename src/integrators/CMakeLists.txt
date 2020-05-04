# Integrators
if (NOT MTS_VERSION)
  message(FATAL_ERROR "Use the top level configuration file")
endif()

macro(add_integrator)
  add_mts_plugin(${ARGN} TYPE integrator)
endmacro()

macro(add_bidir)
  add_integrator(${ARGN} MTS_BIDIR)
endmacro()

if (MTS_CUDA)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52;-std=c++11")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    macro(add_recons)
        add_mtsCUDA_plugin(${ARGN} MTS_BIDIR TYPE integrator)
    endmacro()
    SET(RECONSFILE poisson_solver/BackendCUDA.cu)
else()
    macro(add_recons)
        add_bidir(${ARGN})
    endmacro()
    SET(RECONSFILE poisson_solver/BackendOpenMP.cpp)
endif()

add_integrator(ao       direct/ao.cpp)
add_integrator(direct   direct/direct.cpp)
add_integrator(tangent  direct/tangent.cpp)
add_integrator(path     path/path.cpp)
add_integrator(path_reuse     path/path_reuse.cpp)
add_integrator(volpath  path/volpath.cpp)
add_integrator(volpath_simple path/volpath_simple.cpp)
add_integrator(ptracer  ptracer/ptracer.cpp
                        ptracer/ptracer_proc.h ptracer/ptracer_proc.cpp)

# Photon mapping-based techniques
add_integrator(photonmapper photonmapper/photonmapper.cpp photonmapper/bre.cpp)
add_integrator(ppm          photonmapper/ppm.cpp)
add_integrator(sppm         photonmapper/sppm.cpp)

# Miscellaneous
add_integrator(vpl          vpl/vpl.cpp MTS_HW)
add_integrator(adaptive     misc/adaptive.cpp)
add_integrator(irrcache     misc/irrcache.cpp
                            misc/irrcache_proc.h misc/irrcache_proc.cpp)
add_integrator(multichannel misc/multichannel.cpp)
add_integrator(field        misc/field.cpp)

# Bidirectional techniques
add_bidir(bdpt          bdpt/bdpt.h      bdpt/bdpt.cpp
                        bdpt/bdpt_proc.h bdpt/bdpt_proc.cpp
                        bdpt/bdpt_wr.h   bdpt/bdpt_wr.cpp)

add_bidir(mlt           mlt/mlt.h      mlt/mlt.cpp
                        mlt/mlt_proc.h mlt/mlt_proc.cpp)

add_bidir(erpt          erpt/erpt.h      erpt/erpt.cpp
                        erpt/erpt_proc.h erpt/erpt_proc.cpp)

# SMCMC
add_bidir(spssmlt
        spssmlt/spssmlt.h
        spssmlt/spssmlt.cpp
        spssmlt/spssmlt_sampler.h
        spssmlt/spssmlt_sampler.cpp

        # Algo
        spssmlt/algo/algo.h
        spssmlt/algo/algo_re.h
        spssmlt/algo/algo_classic.h

        # Rendering
        spssmlt/rendering.h
        spssmlt/rendering.cpp

        # Tiles
        spssmlt/tile.h

        # solver
        spssmlt/solver/solver.h
        spssmlt/solver/solver_iterative_reweighted.h
        spssmlt/solver/solver_covariate_grad_log_val.h
        spssmlt/solver/solver_covariate_reference.h

        # For poisson solver
        poisson_solver/Solver.cpp
        poisson_solver/Backend.cpp
        ${RECONSFILE}
        poisson_solver/Defs.cpp)

add_recons(spssmlt_gpt
        spssmlt/spssmlt.h
        spssmlt/spssmlt_gpt.cpp
        spssmlt/spssmlt_sampler.h
        spssmlt/spssmlt_sampler.cpp

        # Rendering
        spssmlt/rendering.h
        spssmlt/rendering.cpp

        # Reconstruction
        poisson_solver/Solver.cpp
        poisson_solver/Backend.cpp
        ${RECONSFILE}
        poisson_solver/Defs.cpp
        )

add_integrator(avg        avg.cpp)
add_bidir(pssmlt
        pssmlt/pssmlt.h
        pssmlt/pssmlt.cpp
        pssmlt/pssmlt_proc.h
        pssmlt/pssmlt_proc.cpp
        pssmlt/pssmlt_sampler.h
        pssmlt/pssmlt_sampler.cpp
        )

add_bidir(erptpss
        erptpss/erptpss.cpp
        erptpss/erptpss.h
        erptpss/erptpss_sampler.h
        erptpss/erptpss_sampler.cpp)

add_recons(gpt
        gpt/gpt.cpp
        gpt/gpt_wr.cpp
        gpt/gpt_proc.cpp

        poisson_solver/Solver.cpp
        poisson_solver/Backend.cpp
        ${RECONSFILE}
        poisson_solver/Defs.cpp

        denoiser/nfor/Regression.cpp
        denoiser/nfor/nfor.cpp
        denoiser/nfor/core/Logging.cpp
        denoiser/nfor/core/thread/ThreadPool.cpp
        denoiser/nfor/core/thread/ThreadUtils.cpp)
