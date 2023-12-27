import os.path as path
import zipfile

def ConcatColumn(originalfilePath, columnFilePath, outputFilePath, delimiter=' '):
    with open(originalfilePath, 'r') as originalFile, open(columnFilePath, 'r') as columnFile, open(
            outputFilePath, 'w') as outputFile:
        for originalLine, columnLine in zip(originalFile, columnFile):
            concatenatedLine = f"{originalLine.strip()}{delimiter}{columnLine.strip()}\n"
            outputFile.write(concatenatedLine)

def CalculateAverageLength(filePath):
    totalLength = 0
    lineCount = 0
    with open(filePath, 'r') as file:
        for line in file:
            totalLength += len(line)
            lineCount += 1
    averageLength = totalLength / lineCount if lineCount > 0 else 0
    return averageLength

def FilterAndWrite(inputFilePath, averageLength, outputFilePath):
    with open(inputFilePath, 'r') as inputFile, open(outputFilePath, 'w') as outputFile:
        for line in inputFile:
            if len(line) <= 2 * averageLength:
                outputFile.write(line)

def CountCommands(line, commandMapping):
    counts = {key: 0 for key in commandMapping}
    words = line.split(' ')
    for word in words:
        command = word.strip()
        if command in counts:
            counts[command] += 1
    return counts

def ProcessInputFile(inputFilePath, outputFilePath, commandMapping):
    with open(inputFilePath, 'r') as inputFile, open(outputFilePath, 'w') as outputFile:
        header = ','.join(commandMapping.keys())
        outputFile.write(f"Line,{header}\n")

        for lineNum, line in enumerate(inputFile, start=1):
            line = line.strip()
            if line:
                counts = CountCommands(line, commandMapping)
                countsStr = ','.join(str(count) for count in counts.values())
                lastWord = line.strip().split()[-1]
                outputFile.write(f"{lineNum},{countsStr},{lastWord}\n")

def ReadMappingFile(filePath):
    commandMapping = {}
    with open(filePath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('=')
                if len(parts) == 2:
                    key, value = parts
                    commandMapping[key.strip()] = int(value.strip())
    return commandMapping

def readApiIndex(filename: str) -> list[str]:
    funciton_names = []
    with open(filename, "r") as f:
        for line in f:
            if len(line) == 0: continue
            parts = line.split("=", 1)
            if len(parts) != 2: continue
            name = parts[0]
            if name.startswith("__"): continue
            funciton_names.append(name)
    return funciton_names

def countApiCallOccurences(filename: str, api_index: list[str]) -> list[dict[str, int]]:
    occurences = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            counts = {}
            for api_call in line.split(" "):
                if api_call not in counts:
                    counts[api_call] = 0
                counts[api_call] += 1
            occurences.append(counts)
    return occurences

def readLabels(filename: str) -> list[str]:
    labels = []
    with open(filename, "r") as f:
        for line in f:
            labels.append(line.strip())
    return labels

def isApiCallUsed(api_counts: list[dict[str, int]], api_name: str) -> bool:
    for counts in api_counts:
        if counts.get(api_name, 0) > 0:
            return True
    return False

def isApiCallCommon(api_counts: list[dict[str, int]], api_name: str) -> bool:
    for counts in api_counts:
        if counts.get(api_name, 0) == 0:
            return False
    return True

def writeApiCounts(filename: str, api_counts: list[dict[str, int]], api_index: list[str]):
    with open(filename, "w") as f:
        f.write(",".join(api_index))
        f.write(",target\n")
        for i, row in enumerate(api_counts):
            for api_call_name in api_index:
                f.write(str(row.get(api_call_name, 0)))
                f.write(",")
            f.write(labels[i])
            f.write("\n")

if __name__ == "__main__":
    initialFolder = "initial"

    original_path = initialFolder + "/all_analysis_data.txt"
    if not path.exists(original_path):
        with zipfile.ZipFile(initialFolder + "/mal-api-2019.zip") as data_zip:
            data_zip.extract("all_analysis_data.txt", initialFolder)

    print("Counting API call occurences...")
    api_index  = readApiIndex(initialFolder + "/ApiIndex.txt")
    labels     = readLabels(initialFolder + "/labels.csv")
    api_counts = countApiCallOccurences(original_path, api_index)

    print("Writing results to file...")
    writeApiCounts("counted.csv", api_counts, api_index)

    print("Writing results to file (skip columns with no calls)...")
    trimmed_api_index = []
    for call_name in api_index:
        if not isApiCallUsed(api_counts, call_name): continue
        trimmed_api_index.append(call_name)

    writeApiCounts("counted-trimmed.csv", api_counts, trimmed_api_index)

    print("Writing results to file (skip common columns)...")
    unique_api_index = []
    for call_name in trimmed_api_index:
        if isApiCallCommon(api_counts, call_name): continue
        unique_api_index.append(call_name)

    writeApiCounts("counted-unique.csv", api_counts, unique_api_index)

    print(f"Done!")

