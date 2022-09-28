
train = []
valid = []
test = []
with open("train_yuan.txt", 'r') as rtrain:
    for fact in rtrain:
        train.append(fact.strip('\n'))
with open("valid_yuan.txt", 'r') as rvalid:
    for fact in rvalid:
        valid.append(fact.strip('\n'))
with open("test_yuan.txt", 'r') as rtest:
    for fact in rtest:
        test.append(fact.strip('\n'))
# print(train)
wtrain = open("train.txt", 'w')
wvalid = open("valid.txt", 'w')
wtest = open("test.txt", 'w')
for fact in train:
    wtrain.write(fact + "\t-1\n")
wtrain.close()
for fact in valid:
    wvalid.write(fact + "\t-1\n")
wvalid.close()
for fact in test:
    wtest.write(fact + "\t-1\n")
wtest.close()