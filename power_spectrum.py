def power_spectrum(dat):

    if dat.ndim == 2:
        p_vec = np.mean(dat,1)
    elif dat.ndim == 3:
        p_vec = np.mean(np.mean(dat,0),1)

    

    plt.plot(p_vec)
    plt.xlim(5, 100)
    plt.show()

    
    ps = np.mean(Sxx[:,:,1:100],2)

    sem = scipy.stats.sem(ps)

    plt.plot(np.arange(1,127),y)
    plt.fill_between(np.arange(1,127),y-sem, y+sem, alpha = 0.5)
    plt.xlim(5,40)
    plt.show()