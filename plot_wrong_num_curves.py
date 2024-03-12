import matplotlib.pyplot as plt
# matplotlib inline

f = plt.figure()
plt.errorbar(
    [0,1,2,3,4],  # X
    [0.359363154,0.887793783,0.80136467,0.648976497,0.627748294], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="GPT-4",
    fmt="r<--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0,1,2,3,4],  # X
    [0.142532221,0.73237301,0.711144807,0.690674754,0.664139], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="gpt-35-turbo",
    fmt="bo-", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/wrong_num_gsm8k.pdf", bbox_inches='tight')


f = plt.figure()
plt.errorbar(
    [0,1,2,3,4],  # X
    [0.413385827,0.622047244,0.665354331,0.669291339,0.712598], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="GPT-4",
    fmt="r<--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0,1,2,3,4],  # X
    [0.299212598,0.401574803,0.437007874,0.377952756,0.405511811], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="gpt-35-turbo",
    fmt="bo-", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/wrong_num_aqua.pdf", bbox_inches='tight')


f = plt.figure()
plt.errorbar(
    [0,1,2,3,4],  # X
    [0.924050633,0.916455696,0.924050633,0.913924051,0.896202], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="GPT-4",
    fmt="r<--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0,1,2,3,4],  # X
    [0.827848101,0.906329114,0.911392405,0.893670886,0.875949367], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="gpt-35-turbo",
    fmt="bo-", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/wrong_num_addsub.pdf", bbox_inches='tight')


f = plt.figure()
plt.errorbar(
    [0,1,2,3,4],  # X
    [0.965,0.978333333,0.973333333,0.965,0.956666667], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="GPT-4",
    fmt="r<--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0,1,2,3,4],  # X
    [0.611666667,0.951666667,0.938333333,0.908333333,0.918333333], # Y
    yerr=[0.0,0.0,0.0,0.0,0.0],     # Y-errors
    label="gpt-35-turbo",
    fmt="bo-", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/wrong_num_multiarith.pdf", bbox_inches='tight')