{
  description = "This is my python dev: Data Science Libraries Edition";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs"; # also valid: "nixpkgs"
  };

  # Flake outputs
  outputs = { self, nixpkgs, }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f:
        nixpkgs.lib.genAttrs allSystems
        (system: f { pkgs = import nixpkgs { inherit system; }; });
    in {
      # Development environment output
      devShells = forAllSystems ({ pkgs }: {
        default = let
          # Use Python 3.11
          python = pkgs.python311;

        in pkgs.mkShell {
          # The Nix packages provided in the environment
          packages = [
            # Python plus helper tools
            (python.withPackages (ps:
              with ps; [
              
                pandas
                matplotlib
                seaborn
                scikit-learn
                pyyaml
                pyvisa pyvisa-py
                scikit-rf
                pyfiglet
                rich # Render rich text, tables, progress bars, syntax highlighting, markdown and more to the terminal

                click # Command Line Interface Creation Kit
                icecream # never use print() to debug again.
                setuptools # Packaging and distributing projects

              ]))
          ];
        };
      });
    };
}
