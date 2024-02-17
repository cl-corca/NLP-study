import matplotlib.pyplot as plt

loss_list =  [10.961938858032227, 4.466469591163794, 3.4835865646834128, 3.3533385156864925, 3.28861841231978, 3.247763253800099, 3.2186196816530397]
plt.plot(range(7),loss_list)
plt.xlabel("Number of epoch")
plt.ylabel("Loss")
plt.title("Transformer: Loss vs Number of epochs")
plt.show()

