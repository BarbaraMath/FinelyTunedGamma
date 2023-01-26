def power_spectrum(ps):
    
    


    p_vec = np.mean(Sxx[1,:,1:100],1)
    p_interpl = scipy.interpolate.interp1d(np.arange(1,127),p_vec)
    
    xnew = np.arange(1,126,0.1)
    print(xnew)

    ynew = p_interpl(xnew)

    plt.plot(p_vec)
    plt.xlim(5,40)
    plt.show()

    print(p_interpl)

    
    ps = np.mean(Sxx[:,:,1:100],2)

    sem = scipy.stats.sem(ps)

    y = np.mean(ps,0)
    plt.plot(np.arange(1,127),y)
    plt.fill_between(np.arange(1,127),y-sem, y+sem, alpha = 0.5)
    plt.xlim(5,40)
    plt.show()