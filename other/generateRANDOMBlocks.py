import numpy as np


def create_RandBlocks(nbBlocks, totalTimeBlocks):
    """ function to generate pseudorandom blocks
    generate N blocks with randomly determined length. Sum of all blocks is equal to totalTimeBLocks.
    inputs:
        nbBlocks: int(number of blocks)
        totalTimeBlocks: int(total time of all blocks)
    output:
        list of blocks to input in labview
    usage:
        lowBlocks = create_RandBlocks(6, 1800)
        highBlocks = create_RandBlocks(6, 1800)"""

    if nbBlocks != 0 and totalTimeBlocks != 0:
        randomBlocks = []  # initiate array
        randList = np.random.rand(nbBlocks)  # generate N random numbers
        sum_randList = sum(randList)  # sum the N rand nums generated
        for i in randList:
            randomBlocks.append(round(i/sum_randList*totalTimeBlocks))  # divide each rand generated number by the sum. this way we ensure that the sum of all generated blocks is equal to 1h or 30min.
        # because of rounding imprecisions we have to check if the sum of the generated blocks is equal to the target.
        # If not it means that we have a few extra or missing seconds. we allocate/remove them randomly from a block.
        if sum(randomBlocks) != totalTimeBlocks:
            leftover = totalTimeBlocks - sum(randomBlocks)
            if leftover > 0:  # if.. means we are short of a few seconds, we add them to a random block
                while leftover != 0:
                    foo = np.random.randint(nbBlocks)
                    randomBlocks[foo] += 1
                    leftover -= 1
            if leftover < 0:  # same except we have too much so we randomly substract.
                while leftover != 0:
                    foo = np.random.randint(nbBlocks)
                    randomBlocks[foo] -= 1
                    leftover += 1
    else:
        print("Error: nbBlocks and totalTimeBlocks should not be equal to 0.")
    return randomBlocks


RNG = np.random.rand(1)  # generate rand number [0:1] to determine if first block is High or Low.
print("Start with:", "High% Block" if RNG > 0.5 else "Low% Block")
lowBlocks = create_RandBlocks(6, 1800)  # generate rand blocks. 6 blocks High, 6 Low, session is 1h, so we want total High and total Low = 1800s.
highBlocks = create_RandBlocks(6, 1800)
print("Low blocks duration", lowBlocks, "total: ", sum(lowBlocks))
print("High blocks duration", highBlocks, "total: ", sum(highBlocks))

sessionBlocks = []  # merge them to make the session.
for i, j in zip(lowBlocks, highBlocks):
    if RNG < 0.5:
        sessionBlocks.append([i, j])
    if RNG > 0.5:
        sessionBlocks.append([j, i])

print("Session: ", sessionBlocks)