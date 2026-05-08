# Numba 0.62.x (PyEventBT) falla con NumPy 2.4+ ("Numba needs NumPy 2.3 or less").
# Este paso se ejecuta despues de requirements.txt para bajar NumPy si pip lo subio
# (p. ej. pip install -U numpy o resolucion antigua sin pin estricto).
function Ensure-NumpyNumbaAbi([string]$PythonExe) {
    & $PythonExe -m pip install "numpy>=1.24,<=2.3.5"
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo al fijar NumPy compatible con Numba (numpy>=1.24,<=2.3.5)"
    }
}
