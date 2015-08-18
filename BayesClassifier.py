__author__ = 'santosh'

import os,sys,math

from nltk.corpus import stopwords

class BayesClassifier:


    def __init__(self):


        self.total = 0

        self.attributeSet = set()


        self.classes = {} #Keep the count of the classes ham and spam
        self.counts = {} #Keep the count of attribute values. how many ham emails contains a particular word

        self.prior = {}

        self.conditional = {}

        self.hamEmails = 0
        self.spamEmails = 0
        self.numberOfHamWords = 0
        self.numberOfSpamWords = 0
        self.classesTF = {}
        self.stopset = set()



        inputFile = open(os.path.join(os.path.expanduser('~'),'trainSpam'),'r')

        ##Create the total set of words in the corpus

        self.stopset = set(stopwords.words('english'))

        for line in inputFile:



            tokens = line.split(" ")[2:]
            tokens = [word for (index,word) in enumerate(tokens) if index%2 == 0]

            #print(tokens)


            for item in tokens:

                if(item not in self.stopset):

                    self.attributeSet.add(item)

            #Clear the tokens before next line
            tokens = []



        print("Total number of attributes : " + str(len(self.attributeSet)))


        inputFile.close()



    def classify(self):

        #testData = open(os.path.join(os.path.expanduser('~'),'testSpam'),'r')
        testData = open(os.path.join(os.path.dirname(__file__),'test'),'r')
        output = open(os.path.join(os.path.dirname(__file__),'resultSpam'),'w')

        groundTruthList = list()
        predictedTruthList = list()

        for line in testData:

            tokens = line.split(" ")

            #print(str(self.numberOfHamWords))



            groundTruth = tokens[1]

            prediction = ""

            wordList = [word for index,word in enumerate(tokens[2:]) if index % 2 ==0]

            probabilityHam = 0.0
            probabilitySpam = 0.0

            for word in wordList:

                if word in self.attributeSet:

                    probabilityHam = probabilityHam + math.log10(self.conditional["ham"][word])
                    probabilitySpam = probabilitySpam + math.log10(self.conditional["spam"][word])

                else:

                    probabilityHam = probabilityHam + math.log10(1/float(self.numberOfHamWords + len(self.attributeSet)))
                    probabilitySpam = probabilitySpam + math.log10(1/float(self.numberOfSpamWords + len(self.attributeSet)))


            prediction = "spam" if probabilitySpam > probabilityHam else "ham"

            groundTruthList.append(groundTruth)
            predictedTruthList.append(prediction)


            output.write(groundTruth + ";" + prediction + "\n")


        self.printSummary(groundTruthList,predictedTruthList)



        testData.close()
        output.close()




    def BinomialAttributes(self):


            ##Iterate through the file once again to populate the counts and classes dictionary

            print("==================== Binomial Naive Bayes =====================\n")

            self.counts.setdefault("ham",{})
            self.counts.setdefault("spam",{})

            for word in self.attributeSet:
                self.counts["ham"].setdefault(word,0)
                self.counts["spam"].setdefault(word,0)


            #tokens.clear()

            #inputFile = open(os.path.join(os.path.expanduser('~'),'trainSpam'),'r')

            inputFile = open(os.path.join(os.path.dirname(__file__),'train'),'r')


            #+++++++++++++++++++++ Binomial Naive Bayes +++++++++++++++++++++++++++++

            for line in inputFile:

                self.total += 1

                tokens = line.split(" ")

                classOfEmail = tokens[1]
                self.classes.setdefault(classOfEmail,0)
                self.classes[classOfEmail] += 1


                #Take all the words which are at even index starting from 2
                for i in range(2,len(tokens),2):

                    if(tokens[i] not in self.stopset):

                        self.counts[classOfEmail][tokens[i]] +=1


            #!!!!!!!!!!!!!!!!!!!!!!!!! Prior Calculation and Conditional Calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


            for word,count in self.counts["ham"].items():
                self.numberOfHamWords += count


            for word,count in self.counts["spam"].items():
                self.numberOfSpamWords += count

           # print(" ham " + str(self.numberOfHamWords))

            #Calculate the prior

            for classOfEmail,count in self.classes.items():
                self.prior.setdefault(classOfEmail,0)
                self.prior[classOfEmail]=float(count/self.total)

            # print("Total ham emails : " + str(self.classes["ham"]))
            # print("Total ham emails : " + str(self.classes["spam"]))
            # print(self.prior)


            #Calculate the conditionals

            self.conditional.setdefault("ham",{})
            self.conditional.setdefault("spam",{})

            for classOfEmail,attribute in self.counts.items():

                #print(attribute)

                for word,count in attribute.items():

                    self.conditional[classOfEmail].setdefault(word,0)
                    self.conditional[classOfEmail][word] = float((count + 1)/float(self.classes[classOfEmail] + len(self.attributeSet)))




    def TermFrequencyAttribute(self):

        print("==================== Multinomial Naive Bayes =====================\n")

        self.counts.setdefault("ham",{})
        self.counts.setdefault("spam",{})

        for word in self.attributeSet:
            self.counts["ham"].setdefault(word,0)
            self.counts["spam"].setdefault(word,0)



        #inputFile = open(os.path.join(os.path.expanduser('~'),'trainSpam'),'r')

        inputFile = open(os.path.join(os.path.dirname(__file__),'train'),'r')

        #@@@@@@@@@@@@@@@@@@@@@@@ Naive Bayes with Multinomial assumption@@@@@@@@@@@@@@@@@@@@@@@@@@@@



        for line in inputFile:

            self.total += 1

            tokens = line.split(" ")

            #print(tokens)

            classOfEmail = tokens[1]
            self.classes.setdefault(classOfEmail,0)
            self.classes[classOfEmail] += 1


            #Take all the words which are at even index starting from 2
            for i in range(2,len(tokens),2):

                if(tokens[i] not in self.stopset):

                    self.counts[classOfEmail][tokens[i]] += int(tokens[i+1])

            #tokens.clear()
            tokens = []

     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Prior Calculation and Conditional Calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for word,count in self.counts["ham"].items():
            self.numberOfHamWords += count


        for word,count in self.counts["spam"].items():
            self.numberOfSpamWords += count


        self.classesTF["spam"] = self.numberOfSpamWords
        self.classesTF["ham"] = self.numberOfHamWords

        # print("Number of ham words " + str(self.numberOfHamWords))
        # print("Number of spam words " + str(self.numberOfSpamWords))



        #Calculate the prior

        for classOfEmail,count in self.classes.items():
            self.prior.setdefault(classOfEmail,0)
            self.prior[classOfEmail]=float(count/self.total)

        # print("Total ham emails : " + str(self.classes["ham"]))
        # print("Total ham emails : " + str(self.classes["spam"]))
        # print(self.prior)


        #Calculate the conditionals

        self.conditional.setdefault("ham",{})
        self.conditional.setdefault("spam",{})

        for classOfEmail,attribute in self.counts.items():

            #print(attribute)

            for word,count in attribute.items():


                self.conditional[classOfEmail].setdefault(word,0)
                self.conditional[classOfEmail][word] = float((count + 1)/float(self.classesTF[classOfEmail] + len(self.attributeSet)))


    def printSummary(self,groundTruthList,predictedTruthList):

        #Calculate Precision:

        truePositives = 0
        falsePositives = 0
        falseNegetives = 0
        trueNegetive = 0

        for ground,prediction in zip(groundTruthList,predictedTruthList):


            if(prediction in "spam" and ground in "spam"):
                truePositives +=1
            elif(prediction in "ham" and ground in "spam"):
                falseNegetives +=1
            elif (prediction in "spam" and ground in "ham"):
                falsePositives +=1
            else:
                trueNegetive += 1


        precision = float(truePositives/float(truePositives+falsePositives)) * 100
        recall = float(truePositives/float(truePositives+falseNegetives)) * 100

        print("\t\t##########Summary#############\n")
        print("Number of true positives : " + str(truePositives))
        print("Number of true negetives : " + str(trueNegetive))
        print("Number of false positives : " + str(falsePositives))
        print("Number of false negetives : " + str(falseNegetives))

        print("Precision : " + str(precision))
        print("Recall : " + str(recall))





if __name__ == "__main__":

    bayes = BayesClassifier()

    bayes.BinomialAttributes()
    bayes.classify()

    bayes.TermFrequencyAttribute()
    bayes.classify()

    #print os.path.dirname(__file__)








