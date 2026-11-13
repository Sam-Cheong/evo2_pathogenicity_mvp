[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$Content
)

$envPath = Join-Path $PSScriptRoot '.env'
if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw 'Create a .env file next to chat.ps1 and set OPENAI_API_KEY.'
}

$keyPattern = '^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(?:"(?<key>[^"]*)"|''(?<key>[^'']*)''|(?<key>[^\s#"'']*))\s*(?:#.*)?$'
$keyEntry = Select-String -LiteralPath $envPath -Pattern $keyPattern -Encoding UTF8 |
    Select-Object -First 1
if (-not $keyEntry -or [string]::IsNullOrWhiteSpace($keyEntry.Matches[0].Groups['key'].Value)) {
    throw 'Set OPENAI_API_KEY in the .env file next to chat.ps1.'
}
$apiKey = $keyEntry.Matches[0].Groups['key'].Value

$requestBody = @{
    model = 'gpt-6.1-terra'
    messages = @(@{ role = 'user'; content = $Content })
    temperature = 0.7
} | ConvertTo-Json -Depth 4 -Compress

$request = @{
    Uri = 'http://178.253.32.244:3000/v1/chat/completions'
    Method = 'Post'
    Headers = @{ Authorization = "Bearer $apiKey" }
    ContentType = 'application/json; charset=utf-8'
    Body = [System.Text.Encoding]::UTF8.GetBytes($requestBody)
    TimeoutSec = 90
    ErrorAction = 'Stop'
}

$response = Invoke-WebRequest @request -UseBasicParsing
[System.Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray())
