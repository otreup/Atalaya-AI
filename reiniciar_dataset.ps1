Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath ".\dataset\atalaya_lecturas.csv") {
  Copy-Item -LiteralPath ".\dataset\atalaya_lecturas.csv" -Destination ".\dataset\respaldo_lecturas_antes_multiclase.csv" -Force
  Write-Host "Copia anterior creada: dataset\respaldo_lecturas_antes_multiclase.csv"
}
"timestamp,sesion,etiqueta,temp_c,humedad_pct,pir,luz,racha_pir,racha_luz,estado_local,clase_predicha,recomendacion,cliente_ip" | Set-Content -LiteralPath ".\dataset\atalaya_lecturas.csv" -Encoding UTF8
Write-Host "Dataset multiclase reiniciado: dataset\atalaya_lecturas.csv"
