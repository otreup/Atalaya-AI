Set-Location -LiteralPath $PSScriptRoot

$adaptador = Get-NetIPConfiguration |
  Where-Object { $_.IPv4Address -and $_.NetAdapter.Status -eq 'Up' } |
  Select-Object -First 1

if (-not $adaptador) {
  Write-Host 'No hay una red WiFi activa en este PC.' -ForegroundColor Red
  exit 1
}

$ip = $adaptador.IPv4Address.IPAddress
Write-Host ''
Write-Host 'RED ACTUAL DEL PC' -ForegroundColor Cyan
Write-Host "Adaptador: $($adaptador.InterfaceAlias)"
Write-Host "IPv4 del PC: $ip" -ForegroundColor Green
Write-Host ''
Write-Host 'Copia esta linea en firmware_esp32_ia.ino:' -ForegroundColor Yellow
Write-Host ('const char* SERVER_URL = "http://{0}:8000/api/lectura";' -f $ip) -ForegroundColor White
Write-Host ''
Write-Host 'Antes de probar:' -ForegroundColor Cyan
Write-Host '1. PC y ESP32 deben estar conectados al MISMO hotspot de 2.4 GHz.'
Write-Host '2. Carga el firmware despues de actualizar SERVER_URL.'
Write-Host '3. Ejecuta .\prediccion_en_vivo.ps1 y deja esa terminal abierta.'
