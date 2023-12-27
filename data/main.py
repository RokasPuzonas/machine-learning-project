import os


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
        header = ' '.join(commandMapping.keys())
        outputFile.write(f"Line {header}\n")
        
        for lineNum, line in enumerate(inputFile, start=1):
            line = line.strip()
            if line:
                counts = CountCommands(line, commandMapping)
                countsStr = ' '.join(str(count) for count in counts.values())
                lastWord = line.strip().split()[-1]
                outputFile.write(f"{lineNum} {countsStr} {lastWord}\n")


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


if __name__ == "__main__":
    original = 'all_analysis_data.txt'
    columns = 'labels.csv'
    output1 = 'output.csv'
    output2 = 'output2.csv'
    ConcatColumn(original, columns, output1)
    averageLength = CalculateAverageLength(output1)

    FilterAndWrite(output1, averageLength, output2)
    mappingFilePath = 'ApiIndex.txt'
    commandMapping = ReadMappingFile(mappingFilePath)
    finalOutput = 'data.csv'
    ProcessInputFile(output2, finalOutput, commandMapping)
    os.remove("output.csv")
    os.remove("output2.csv")
    print(f"Done")
