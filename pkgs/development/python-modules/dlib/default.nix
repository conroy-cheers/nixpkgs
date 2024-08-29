{
  stdenv,
  buildPythonPackage,
  dlib,
  python,
  pytestCheckHook,
  more-itertools,
  sse4Support ? stdenv.hostPlatform.sse4_1Support,
  avxSupport ? stdenv.hostPlatform.avxSupport,
}:

buildPythonPackage {
  inherit (dlib)
    stdenv
    pname
    version
    src
    nativeBuildInputs
    buildInputs
    cmakeFlags
    passthru
    meta
    ;

  format = "setuptools";

  patches = [ ./build-cores.patch ];

  nativeCheckInputs = [
    pytestCheckHook
    more-itertools
  ];

  postPatch = ''
    substituteInPlace setup.py \
      --replace "more-itertools<6.0.0" "more-itertools" \
      --replace "pytest==3.8" "pytest"
  '';

  setupPyBuildFlags = [
    "--set USE_SSE4_INSTRUCTIONS=${if sse4Support then "yes" else "no"}"
    "--set USE_AVX_INSTRUCTIONS=${if avxSupport then "yes" else "no"}"
  ];
  # Pass CMake flags through to the build script
  preConfigure = ''
    for flag in $cmakeFlags; do
      if [[ "$flag" == -D* ]]; then
        setupPyBuildFlags+=" --set ''${flag#-D}"
      fi
    done
  '';

  dontUseCmakeConfigure = true;

  doCheck =
    !(
      # The tests attempt to use CUDA on the build platform.
      # https://github.com/NixOS/nixpkgs/issues/225912
      dlib.cudaSupport

      # although AVX can be enabled, we never test with it. Some Hydra machines
      # fail because of this, however their build results are probably used on hardware
      # with AVX support.
      || dlib.avxSupport
    );
}
