{
  lib,
  stdenv,
  python,
  buildPythonPackage,
  pythonRelaxDepsHook,
  fetchFromGitHub,
  fetchpatch,
  symlinkJoin,
  autoAddDriverRunpath,

  # build system
  packaging,
  setuptools,
  wheel,

  # dependencies
  which,
  ninja,
  cmake,
  setuptools-scm,
  torch,
  outlines,
  psutil,
  ray,
  pandas,
  pyarrow,
  sentencepiece,
  numpy,
  transformers,
  xformers,
  fastapi,
  uvicorn,
  pydantic,
  aioprometheus,
  pynvml,
  openai,
  pyzmq,
  tiktoken,
  torchvision,
  py-cpuinfo,
  lm-format-enforcer,
  prometheus-fastapi-instrumentator,
  cupy,
  gguf,
  einops,
  importlib-metadata,
  partial-json-parser,
  compressed-tensors,
  mistral-common,
  msgspec,
  numactl,
  tokenizers,
  oneDNN,
  bitsandbytes,

  config,

  cudaSupport ? config.cudaSupport,
  cudaPackages ? { },

  rocmSupport ? false,
  rocmPackages ? { },
  gpuTargets ? [ ],
}@args:

let
  shouldUsePkg = pkg: if pkg != null && lib.meta.availableOn stdenv.hostPlatform pkg then pkg else null;
  
  cutlass = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "refs/tags/v3.5.1";
    sha256 = "sha256-sTGYN+bjtEqQ7Ootr/wvx3P9f8MCDSSj3qyCWjfdLEA=";
  };

  vllm-flash-attn = stdenv.mkDerivation {
    pname = "vllm-flash-attn";
    version = "2.6.2-unstable-2024-10-26";

    src = fetchFromGitHub {
      owner = "vllm-project";
      repo = "flash-attention";
      rev = "5259c586c403a4e4d8bf69973c159b40cc346fb9";
      sha256 = "sha256-6/pDTyP1wy9PSmWtKbmf5EBfdz6eLabhpoA06rfvp+U=";
      fetchSubmodules = true;
      leaveDotGit = true;
    };

    patches = [
      # CUDA arch 8.7 support for Orin devices
      (fetchpatch {
        url = "https://github.com/vllm-project/flash-attention/commit/0aedbfea9b2c39b61a8fa994af1c42a88a2113e4.patch";
        hash = "sha256-FUl4g0WA3Qz4kBam7aSsAo7hIfPZNy6xm3l4qITdNXg=";
      })
    ];

    dontConfigure = true;

    installPhase = ''
      cp -rva . $out
    '';
  };

  cpuSupport = !cudaSupport && !rocmSupport;

  isCudaJetson = cudaSupport && cudaPackages.cudaFlags.isJetsonBuild;

  mergedCudaLibraries = with cudaPackages; [
    cuda_cudart # cuda_runtime.h, -lcudart
    cuda_cccl
    libcusparse # cusparse.h
    libcusolver # cusolverDn.h
    cuda_nvtx
    cuda_nvrtc
    libcublas
  ];

  # Some packages are not available on all platforms
  nccl = shouldUsePkg (cudaPackages.nccl or null);

  getAllOutputs = p: [
    (lib.getBin p)
    (lib.getLib p)
    (lib.getDev p)
  ];

in

buildPythonPackage rec {
  pname = "vllm";
  version = "0.6.4.dev";
  pyproject = true;

  stdenv = if cudaSupport then cudaPackages.backendStdenv else args.stdenv;

  src = fetchFromGitHub {
    owner = "vllm-project";
    repo = "vllm";
    rev = "2f0a0a17a47436fe9709462dfee3bb9d2f91e0a0";
    hash = "sha256-MmsEKtXn0qqPGpBkMV5olo1flqThUAZPLSp9K4RXcMw=";
  };

  patches = [
    ./0001-setup.py-don-t-ask-for-hipcc-version.patch
    ./0002-setup.py-nix-support-respect-cmakeFlags.patch
  ];

  postPatch = ''
    # Ignore the python version check because it hard-codes minor versions and
    # lags behind `ray`'s python interpreter support
    substituteInPlace CMakeLists.txt \
      --replace-fail \
        'set(PYTHON_SUPPORTED_VERSIONS' \
        'set(PYTHON_SUPPORTED_VERSIONS "${lib.versions.majorMinor python.version}"'

    # Pass through PYTHONPATH to worker processes
    substituteInPlace vllm/model_executor/models/registry.py \
      --replace-fail \
        'subprocess.run(' \
        'subprocess.run(env={"PYTHONPATH": ":".join(sys.path)}, args=\'

    # Relax torch dependency manually because the nonstandard requirements format
    # is not caught by pythonRelaxDeps
    substituteInPlace requirements*.txt pyproject.toml \
      --replace-warn 'torch==2.5.1' 'torch==${lib.getVersion torch}' \
      --replace-warn 'torch == 2.5.1' 'torch == ${lib.getVersion torch}'
  '' + lib.optionalString (nccl == null) ''
    # On platforms where NCCL is not supported (e.g. Jetson), substitute Gloo (provided by Torch)
    substituteInPlace vllm/distributed/parallel_state.py \
      --replace-fail '"nccl"' '"gloo"'
  '';

  nativeBuildInputs = [
    cmake
    ninja
    pythonRelaxDepsHook
    which
  ] ++ lib.optionals rocmSupport [
    rocmPackages.hipcc
  ] ++ lib.optionals cudaSupport [
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ] ++ lib.optionals isCudaJetson [
    cudaPackages.autoAddCudaCompatRunpath
  ];


  build-system = [
    packaging
    setuptools
    wheel
  ];

  buildInputs =
    [
      setuptools-scm
      torch
    ]
    ++ (lib.optionals cpuSupport ([
      numactl
      oneDNN
    ]))
    ++ (lib.optionals cudaSupport mergedCudaLibraries ++ (with cudaPackages; [
      nccl
      cudnn
      libcufile
    ]))
    ++ (lib.optionals rocmSupport (
      with rocmPackages;
      [
        clr
        rocthrust
        rocprim
        hipsparse
        hipblas
      ]
    ));

  dependencies =
    [
      aioprometheus
      fastapi
      lm-format-enforcer
      numpy
      openai
      outlines
      pandas
      prometheus-fastapi-instrumentator
      psutil
      py-cpuinfo
      pyarrow
      pydantic
      pyzmq
      ray
      sentencepiece
      tiktoken
      tokenizers
      msgspec
      gguf
      einops
      importlib-metadata
      partial-json-parser
      compressed-tensors
      mistral-common
      bitsandbytes
      torch
      torchvision
      transformers
      uvicorn
      xformers
    ]
    ++ uvicorn.optional-dependencies.standard
    ++ aioprometheus.optional-dependencies.starlette
    ++ lib.optionals cudaSupport [
      cupy
      pynvml
    ];

  dontUseCmakeConfigure = true;
  cmakeFlags = [
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_CUTLASS" "${lib.getDev cutlass}")
    (lib.cmakeFeature "VLLM_FLASH_ATTN_SRC_DIR" "${lib.getDev vllm-flash-attn}")
  ] ++ lib.optionals cudaSupport [
    (lib.cmakeFeature "TORCH_CUDA_ARCH_LIST" "${torch.gpuTargetString}")
    (lib.cmakeFeature "CUTLASS_NVCC_ARCHS_ENABLED" "${cudaPackages.cudaFlags.cmakeCudaArchitecturesString}")
    (lib.cmakeFeature "CUDA_TOOLKIT_ROOT_DIR" "${symlinkJoin {
      name = "cuda-merged-${cudaPackages.cudaVersion}";
      paths = builtins.concatMap getAllOutputs mergedCudaLibraries;
    }}")
    (lib.cmakeFeature "CAFFE2_USE_CUDNN" "ON")
    (lib.cmakeFeature "CAFFE2_USE_CUFILE" "ON")
    (lib.cmakeFeature "CUTLASS_ENABLE_CUBLAS" "ON")
  ] ++ lib.optionals cpuSupport [
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_ONEDNN" "${lib.getDev oneDNN}")
  ];

  env =
    lib.optionalAttrs cudaSupport {
      VLLM_TARGET_DEVICE = "cuda";
      CUDA_HOME = "${lib.getDev cudaPackages.cuda_nvcc}";
    }
    // lib.optionalAttrs rocmSupport {
      VLLM_TARGET_DEVICE = "rocm";
      # Otherwise it tries to enumerate host supported ROCM gfx archs, and that is not possible due to sandboxing.
      PYTORCH_ROCM_ARCH = lib.strings.concatStringsSep ";" rocmPackages.clr.gpuTargets;
      ROCM_HOME = "${rocmPackages.clr}";
    }
    // lib.optionalAttrs cpuSupport {
      VLLM_TARGET_DEVICE = "cpu";
    };

  pythonRelaxDeps = true;

  # pythonImportsCheck = [ "vllm" ];

  meta = with lib; {
    description = "High-throughput and memory-efficient inference and serving engine for LLMs";
    changelog = "https://github.com/vllm-project/vllm/releases/tag/v${version}";
    homepage = "https://github.com/vllm-project/vllm";
    license = licenses.asl20;
    maintainers = with maintainers; [
      happysalada
      lach
    ];

    # CPU support relies on unpackaged dependency `intel_extension_for_pytorch`
    broken = cpuSupport;
  };
}
