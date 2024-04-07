import matplotlib.pyplot as plt
class Logger_res(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            self.file = open(fpath, 'w')

    def append(self, line):
        self.file.write(line+'\n')
        self.file.flush()

    # def plot(self, names=None):
    #     names = self.names if names == None else names
    #     numbers = self.numbers
    #     for _, name in enumerate(names):
    #         x = np.arange(len(numbers[name]))
    #         plt.plot(x, np.asarray(numbers[name]))
    #     plt.legend([self.title + '(' + name + ')' for name in names])
    #     plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

if __name__=='__main__':
    logger_res=Logger_res('log.txt')
    logger_res.append(names)
    logger_res.append(Results)
    logger_res.append(Results.replace('|','').replace('/','\t'))