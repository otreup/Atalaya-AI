Set-Location -LiteralPath $PSScriptRoot
$modelPath = Join-Path $PSScriptRoot "modelos\atalaya_modelo.joblib"
if (-not (Test-Path -LiteralPath $modelPath)) {
  Write-Host "No existe modelos\atalaya_modelo.joblib." -ForegroundColor Yellow
  Write-Host "Primero completa el dataset y ejecuta .\entrenar_modelo.ps1." -ForegroundColor Yellow
  exit 1
}

python .\receiver.py --label 0 --sesion prueba_en_vivo --model $modelPath --no-save
