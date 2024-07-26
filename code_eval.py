import re


class _TreeNode:
    def __init__(self):
        self.label = ""
        self.code = ""
        self.children = []

    def insert(self, code, label):
        branch = 0
        if code[0].isdigit():
            branch = int(code[0])
        else:
            branch = ord(code[0]) - ord('a') + 10

        if len(code) > 1:
            code = code[1:]
            self.children[branch].insert(code, label)
        else:
            a = _TreeNode()
            a.label = label
            a.code = code
            self.children += [a]

    def Print(self, depth):
        for i in range(depth):
            print('\t')
        print(self.code, self.label)
        for c in self.children:
            c.Print(depth + 1)

    def getChoicesAlongPath(self, path):
        branch = 0
        if path[0].isdigit():
            branch = int(path[0])
        else:
            branch = ord(path[0]) - ord('a') + 10

        if (len(self.children)) <= branch:
            if branch == 0:
                if len(path) > 1:
                    return [1] + self.getChoicesAlongPath(path[1:])
                else:
                    return [1]

        if len(path) > 1:
            return [len(self.children)] + self.children[
                branch].getChoicesAlongPath(path[1:])
        else:
            return [len(self.children)]

    def CodeToText(self, code):
        branch = 0
        if code[0] != '*':
            if code[0].isdigit():
                branch = int(code[0])
            else:
                branch = ord(code[0]) - ord('a') + 10

            if (len(self.children)) <= branch:
                if branch == 0:
                    if len(code) > 1:
                        return ["unspecified"] + self.CodeToText(code[1:])
                    else:
                        return ["unspecified"]

            if len(code) > 1:
                return [self.children[branch].label] + self.children[
                    branch].CodeToText(code[1:])
            else:
                return [self.children[branch].label]
        else:
            return ["*"]


class _Axis:
    def __init__(self):
        self.tree = _TreeNode()

    def Print(self):
        self.tree.Print(0)

    def getChoicesAlongPath(self, path):
        return self.tree.getChoicesAlongPath(path)

    def CodeToText(self, code):
        return self.tree.CodeToText(code)


class code:
    def __init__(self, filename):
        self.axes = []
        ifs = open(filename, 'r')

        lines = list(map(lambda line: re.sub("\n", "", line), ifs.readlines()))

        ifs.close()
        lineNo = 0
        while lineNo < len(lines):
            l = lines[lineNo]
            lineNo += 1
            #            print lineNo,l
            if l != "":
                if l.startswith("**"):
                    a = _Axis()
                    tok = l.split()
                    a.tree.label = tok[1]
                    self.axes += [a]
                else:
                    code = l[l.find("[") + 1:l.find("]")]
                    label = l[l.find("]") + 2:]
                    self.axes[-1].tree.insert(code, label)

    def Print(self):
        for a in self.axes:
            a.Print()

    def CodeToText(self, code):
        result = []
        code = code.split("-")
        for ax in range(len(code)):
            result += [self.axes[ax].CodeToText(code[ax])]
        return result

    def evaluate(self, correct, coding, verbose=False):
        # settings
        unspecifiedW = 0.5
        wrongW = 1.0
        correctW = 0.0

        correct = correct.split("-")
        classified = coding.split("-")
        # print "classified as %s, correct is %s" % (classified,correct)

        errorcount = 0.0
        for ax in range(len(correct)):

            if verbose:
                print("[%s - %s]" % (correct[ax], classified[ax]))

            axisMaximalError = 0.0
            axisErrorCount = 0.0

            cor = correct[ax]
            if cor[0] != 'C':
                axis = self.axes[ax]
                choices = axis.getChoicesAlongPath(cor)

                # set branching factor to a constant value (for demonstration purposes)
                # choices=len(choices)*[10]

                wrong = False
                unspec = False
                for i in range(len(cor)):
                    # branching factor 2
                    # choices[i]=2
                    # print "B=",choices[i],
                    localE = (1.0 / (i + 1)) * (1.0 / choices[i])

                    # print localE,
                    axisMaximalError += localE * wrongW

                    # print cor[i], classified[ax][i],wrong,unspec,
                    if cor[i] == classified[ax][
                            i] and not wrong and not unspec:
                        axisErrorCount += localE * correctW
                    elif cor[i] == '0' and classified[ax][
                            i] == '*' and not wrong:
                        axisErrorCount += localE * correctW
                        unspec = True
                    elif (classified[ax][i] == '*' and not wrong) or unspec:
                        axisErrorCount += localE * unspecifiedW
                        unspec = True
                    else:
                        wrong = True
                        axisErrorCount += localE * wrongW

                normAxisError = axisErrorCount / axisMaximalError
                if verbose:
                    print(normAxisError)

                errorcount += 0.25 * normAxisError

        if verbose:
            print(" error count: %f " % errorcount)

        return errorcount