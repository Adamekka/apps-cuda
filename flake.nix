{
  description = "CUDA development shell (flake)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    ,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          pure = true;
          name = "cuda-env-shell";

          packages = with pkgs; [
            git
            gitRepo
            gnupg
            autoconf
            curl
            pkgconf
            (opencv.override {
              enableGtk2 = true;
              gtk2 = pkgs.gtk2;
            })
            procps
            gnumake
            util-linux
            m4
            gperf
            unzip
            linuxPackages.nvidia_x11
            cudaPackages_12_8.cudatoolkit
            libGLU
            libGL
            clang
            ccls
            xorg.libXi
            xorg.libXmu
            freeglut
            xorg.libXext
            xorg.libX11
            xorg.libXv
            xorg.libXrandr
            zlib
            ncurses5
            binutils
            stdenv.cc
            pkg-config
          ];

          shellHook = ''
            if grep -qi microsoft /proc/version 2>/dev/null || [ -n "$WSL_INTEROP" ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib"
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
              export EXTRA_CCFLAGS="-I/usr/include"
            else
              export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib"
              export EXTRA_LDFLAGS="-L${pkgs.linuxPackages.nvidia_x11}/lib"
              unset EXTRA_CCFLAGS
            fi

            export CUDA_PATH=${pkgs.cudaPackages_12_8.cudatoolkit}
          '';
        };
      }
    );
}
