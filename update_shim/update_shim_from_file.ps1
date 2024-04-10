# This script updates the shim settings of Siemens MRI systems using the 'AdjValidate' function.
# It reads the current shim settings from the MRI system.
# It reads the file shim_update_file.txt with the corresponding delta values to update the shim channels.
# It reads the file fre_update_file.txt with the corresponding delta values to update the center frequency.
# It updates the shim accordingly and updates transmit voltage and center frequency to prevent any automatic shim procedures from the MRI System.  
#
# Make sure the function 'AdjValidate' is installed on the MRI system!!! (cp AdjValidate.exe to C:\ProgramData\Siemens\Numaris\MriCustomer\bin )
# (from C:\Program Files\Siemens\Numaris\Mars\MriCustomer\bin ???)
#
# MIT License 
# Copyright 2024 Niklas Wehkamp
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# Function to capture shim values from command-line output and extract decimal values
function Get-DecimalShimOutput ($command) {
    $output = Invoke-Expression $command
    # Find the line containing the decimal values
    $decimalValuesLine = $output | Select-String -Pattern '(([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?( ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?)+)'
    if ($decimalValuesLine) {
        # Extract decimal values from the line
        $decimalValues = $decimalValuesLine -split '\s+' | Where-Object { $_ -match '\d+(\.\d+)?' }
        # Convert decimal values to decimal type
        $decimalArray = $decimalValues | ForEach-Object { [decimal]$_ }
        return $decimalArray
    } else {
        Write-Host "No shim values found in the command output."
        #return @()
    }
}

# Function to capture Transmit Voltage command-line output and extract decimal values
function Get-DecimalTraOutput ($command) {
    $output = Invoke-Expression $command
    # Find the line containing the decimal values
    $decimalValuesLine = $output | Select-String -Pattern '[0-9]*\.[0-9]+'
    if ($decimalValuesLine) {
        # Extract decimal values from the line
        $decimalValues = $decimalValuesLine -split '\s+' | Where-Object { $_ -match '\d+(\.\d+)?' }
        # Convert decimal values to decimal type
        $decimalArray = $decimalValues | ForEach-Object { [decimal]$_ }
        return $decimalArray
    } else {
        Write-Host "No transmit voltage values found in the command output."
        #return @()
    }
}

# Function to capture center frequency from command-line output and extract decimal values
function Get-DecimalFreOutput ($command) {
    $output = Invoke-Expression $command
    # Find the line containing the decimal values
    $decimalValuesLine = $output | Select-String -Pattern '[0-9]+'
    if ($decimalValuesLine) {
        # Extract decimal values from the line
        $decimalValues = $decimalValuesLine -split '\s+' | Where-Object { $_ -match '\d+(\.\d+)?' }
        $lastdecimalValues = $decimalValues[-1]
        # Convert decimal values to decimal type
        $decimalArray = $lastdecimalValues | ForEach-Object { [decimal]$_ }
        return $decimalArray
    } else {
        Write-Host "No center frequency values found in the command output."
        #return @()
    }
}

# Function to read an array of decimal variables from a file
function Read-DecimalArrayFromFile ($filePath) {
    $content = Get-Content $filePath
    $decimalArray = @()
    foreach ($value in $content) {
        $splitValues = $value.Trim().Split(' ')
        foreach ($splitValue in $splitValues) {
            $decimalValue = [decimal]$splitValue
            $decimalArray += $decimalValue
        }
    }
    return $decimalArray
}

# Function to read an integer variable from a file
function Read-IntegerFromFile ($filePath) {
    return Get-Content $filePath | ForEach-Object { [int]$_ }
}



# Main script starts here:
# Capture command-line output into variables
$fre_get = Get-DecimalFreOutput "adjvalidate -fre -get"
$tra_get = Get-DecimalTraOutput "adjvalidate -tra -get"
$shim_get = Get-DecimalShimOutput "adjvalidate -shim -get -mp"

# Read values from file
$shim_update = Read-DecimalArrayFromFile "shim_update_file.txt"
$fre_update = Read-IntegerFromFile "fre_update_file.txt"

##
# Calculate new shim values
$new_shim = @()
for($count=0;$count -lt $shim_get.count;$count++){
    $new_shim += $shim_get[$count] + $shim_update[$count]
}

# Calculate new center frequency
$new_fre = if ($fre_get -ne $null -and $fre_update -ne $null) { $fre_get + $fre_update } else { $fre_get }

## Set new values
#Set the shim currents
#$shimValuesString = $new_shim -join " "
#& "adjvalidate" "-shim" "-set" "-mp" "$shimValuesString"
& "adjvalidate" "-shim" "-set" "-mp" $new_shim[0] $new_shim[1] $new_shim[2] $new_shim[3] $new_shim[4] $new_shim[5] $new_shim[6] $new_shim[7]
#Set the transmit voltage
& "adjvalidate" "-tra" "-set" $tra_get
#Set the center frequency
& "adjvalidate" "-fre" "-set" "$new_fre"

# Output values
Write-Host "fre_get: $fre_get"
Write-Host "tra_get: $tra_get"
Write-Host "shim_get: $($shim_get -join ' ')"
Write-Host "fre_update: $fre_update"
Write-Host "shim_update: $($shim_update -join ' ')"
Write-Host "new_fre: $new_fre"
Write-Host "new_shim: $($new_shim -join ' ')"

# End of script action
