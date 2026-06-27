with import <nixpkgs> {};
mkShell {
    buildInputs = [
        (python3.withPackages (ps: with ps; [
            black
            flake8
            matplotlib
            pandas
            scikit-learn
            scipy
            statsmodels
        ]))
    ];
    shellHook = ''
    '';
}
