with import <nixpkgs> {};
mkShell {
    buildInputs = [
        (python314.withPackages (ps: with ps; [
            black
            flake8
            matplotlib
            pandas
            requests
            seaborn
        ]))
    ];
    shellHook = ''
        export NIX_ENFORCE_NO_NATIVE="0"
    '';
}
