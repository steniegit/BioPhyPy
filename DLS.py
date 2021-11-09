class DLS_Data():
    
    def __init__(self, folder='', verbose=False, labels={}, pfolder=''):
        '''
        Initialize dls data
        folder: Folder with dat files
        labels: dictionary with labels (optional for plotting)
        pfolder: folder with pictures (optional)
        '''
        self.folder = folder
        self.labels = labels
        self.pfolder = pfolder
        # Load data and do outlier rejection
        self.load(verbose=verbose)
        self.reject_incomplete()
        return None
    
    def load(self, verbose=False):
        # Find files in folder
        fns = glob.glob(self.folder + '/*.dat')
        print("Found %i files in folder %s" % (len(fns), self.folder))
        fns.sort()
        # Create list for dictionary
        run, acf, acf_x, fit, fit_x, dis, dis_x, pos, spos, row, col, pics = [], [], [], [], [], [], [], [], [], [], [], []
        for fn in fns:
            # Extract position
            pos_ = fn.split('.')[0].split('-')[-1]
            pfns_ = self.pfolder + '/' + pos_ + '-0.png'
            if os.path.isfile(pfns_):
                #print("Picture %s found!" % pfns_)
                pics.append(pfns_)
            else:
                pics.append('')
            # Create second position with zero filling (easier for sorting later)
            spos_ = pos_[0] + pos_[1:].zfill(2)
            # Row, columns and run numbers
            row_ = pos_[0]
            col_ = int(pos_[1:])
            run_ = int(fn.split('.')[1].split('-')[1])
            with open(fn, 'r') as f:
                # Get acf
                acf_ = list(itertools.takewhile(lambda x: '&' not in x, 
                                                itertools.dropwhile(lambda x: '#acf0' not in x, f)))
                # Get fit 
                fit_ = list(itertools.takewhile(lambda x: '&' not in x, 
                                                itertools.dropwhile(lambda x: '#fit0' not in x, f)))
                # Get dis0 
                dis_ = list(itertools.takewhile(lambda x: '&' not in x, 
                                                itertools.dropwhile(lambda x: '#dis0' not in x, f)))
            acf_ = np.array([list(map(float, b.split())) for b in acf_[1:]])
            fit_ = np.array([list(map(float, b.split())) for b in fit_[1:]])
            dis_ = np.array([list(map(float, b.split())) for b in dis_[1:]])
            # Fill lists
            run.append(run_)
            acf.append(acf_)
            #acf_x = acf_[:,0]
            fit.append(fit_)
            #fit_x = fit_[:,0]
            dis.append(dis_)
            #dis_x = dis_[:,0]
            pos.append(pos_)
            row.append(row_)
            col.append(col_)
            spos.append(spos_)
        # Sort indices based on column, row and runnr
        inds = np.lexsort((run, spos)) #np.argsort(spos_)
        # Fill class 
        self.fns = [fns[i] for i in inds]
        self.run = [run[i] for i in inds]
        self.acf = [acf[i] for i in inds]
        self.fit = [fit[i] for i in inds]
        self.dis = [dis[i] for i in inds]
        self.pos = [pos[i] for i in inds]
        self.spos = [spos[i] for i in inds]
        self.row = [row[i] for i in inds]
        self.col = [col[i] for i in inds]
        self.keep = np.ones(len(self.pos))
        self.out = np.zeros(len(self.pos))
        self.pics = pics
        return None
    
    def reject_incomplete(self, verbose=False):
        '''
        This is to remove incomplete acf scans and generates
        arrays for acfs, fits and diss.
        The x-values are now stored in acf_x, fit_x and dis_x
        This makes averaging, plotting etc. easier
        '''
        # Get list with lengths
        acf_size = []
        for i in self.acf:
            acf_size.append(len(i))
        acf_size = np.array(acf_size)
        self.acf_size = acf_size
        # Determine median and throw out all non-median values
        size_median = int(np.median(self.acf_size))
        indices = np.argwhere(self.acf_size == size_median).squeeze()
        out_ind = np.argwhere(self.acf_size != size_median).squeeze()
        if verbose:
            print("Remove %i out of %i entries due to inconsistent acf length" % (len(self.run) - len(indices), len(self.run)))
            for index in out_ind:
                print(self.fns[index])
        # Modify entries
        self.run   = np.array([self.run[index] for index in indices])
        self.acf_x = np.array(self.acf[0][:,0])
        self.acf   = np.array([self.acf[index][:,1] for index in indices]).T
        self.fit   = [self.fit[index] for index in indices]
        #self.fit_x = np.array(self.fit[0][:,0])
        #self.fit   = np.array([self.fit[index][:,1] for index in indices])
        self.dis_x = np.array(self.dis[0][:,0])
        self.dis   = [self.dis[index] for index in indices]
        self.pos   = np.array([self.pos[index] for index in indices])
        self.spos  = np.array([self.spos[index] for index in indices])
        self.row   = np.array([self.row[index] for index in indices])
        self.col   = np.array([self.col[index] for index in indices])
        self.fns   = np.array([self.fns[index] for index in indices])
        self.keep  = self.keep[indices]
        self.out   = self.out[indices]
        self.pics  = np.array([self.pics[index] for index in indices])
        return None

    def reject_outliers_acf(self, tol='3percent', verbose=False):
        '''
        Outlier rejection based on acf 
        @param tol: The tolerance cutoff. Can be 'XXpercent' or 'XXunits', where XX is a number in any 
        format. With 'percent', the cutoff is defined as a percentage of the median, with 'units' it is
        absolute. All exposures for which the q-averaged scattering falls further than one cutoff 
        distance from the median are rejected.
        '''
        # Tolerance factor
        if 'percent' in tol:
            tol_fact = float(tol.split('percent')[0]) * .01
        else:
            raise Exception('Bad input')
        
        # Loop through positions
        for spos in np.unique(self.spos):
            # Use pos internally
            pos = self.pos[self.spos == spos][0]
            # calculate median and tolerances, here we take mean
            mean = np.median(self.acf[:, self.pos==pos], axis=1).reshape((-1,1)) 
            # work out which repeats to keep:
            # keep = np.arange(len(statistics))[np.abs(statistics-mean) < tolerance]
            keep = np.sum(np.abs(self.acf[:,self.pos==pos]-mean), axis=0) < tol_fact * np.sum(np.abs(mean))
            self.keep[self.pos==pos] = keep
            self.out[self.pos==pos] = np.invert(keep)
            if verbose:
                print('%s: RejectS() rejected %u out of %u repeats (%.0f%%)'%(pos, np.sum(self.pos==pos)-np.sum(keep), np.sum(self.pos==pos), float(np.sum(self.pos==pos)-np.sum(keep))/np.sum(self.pos==pos)*100))
        self.keep = self.keep.astype('int')
        self.out = self.out.astype('int')
        return None

    def average_pos(self, plot=False, plotall=False, savefig=False):
        '''
        Averages acf for each position
        plot: Plot results
        plotall: Even plot positions averaging cannot be done (e.g. only one spectrum or only outliers)
        savefig: saves each position as pos.pdf
        '''
        acf_average, pos_average = [], []
        for spos in np.unique(self.spos):
            pos = self.pos[self.spos==spos][0]
            if np.sum(self.pos==pos) < 2:
                print('%s: Less than 2 spectra available (before outlier rejection). Cannot do averaging' % pos)
                if plotall:
                    # Plot
                    fig, ax = plt.subplots(1)
                    if np.sum(self.keep[self.pos==pos]) > 0:
                        h1 = ax.semilogx(self.acf_x, sub_in, label='Keep (%i)' % np.sum(self.keep[self.pos==pos]), color='green', lw=.5, alpha=.5)
                    if np.sum(self.out[self.pos==pos]) > 0:
                        h3 = ax.semilogx(self.acf_x, sub_out, label='Out (%i)' % np.sum(self.out[self.pos==pos]), color='red', lw=.5, alpha=.5)
                    ax.legend()
                    fig.savefig(pos + '_avg.pdf')
                continue
            else:
                if np.sum(self.keep[self.pos==pos])>1:
                    sub = self.acf[:,self.pos==pos]
                    pic = self.pics[self.pos==pos][0]
                    sub_in = sub[:, np.argwhere(self.keep[self.pos==pos]).squeeze()]
                    sub_out = sub[:, np.argwhere(self.out[self.pos==pos]).squeeze()]
                    sub_av = np.average(sub_in, axis=1)
                    acf_average.append(sub_av)
                    pos_average.append(pos)
                    if plot:
                        # Plot
                        fig, ax = plt.subplots(1)
                        h1 = ax.semilogx(self.acf_x, sub_in, label='Keep (%i)' % np.sum(self.keep[self.pos==pos]), color='green', lw=.5, alpha=.5)
                        h2 = ax.semilogx(self.acf_x, acf_average[-1], label='Average', color='blue', lw=2, alpha=.5)
                        if np.sum(self.out[self.pos==pos]) > 0:
                            h3 = ax.semilogx(self.acf_x, sub_out, label='Out (%i)' % np.sum(self.out[self.pos==pos]), color='red', lw=.5, alpha=.5)
                            ax.legend(handles=[h1[0], h3[0], h2[0]])
                        else:
                            ax.legend(handles=[h1[0],  h2[0]])
                        ax.set_xlabel('Time / s')
                        ax.set_ylabel('ACF')
                        ax.set_xlim([np.min(self.acf_x), np.max(self.acf_x)])
                        try:
                            label = pos + ': ' +self.labels[pos]
                        except:
                            label = pos 
                        ax.set_title(label)
                        if len(pic) > 0:
                            pic = mpimg.imread(pic)
                            ax_pic = fig.add_axes([.3, .46, .4, .4])
                            ax_pic.imshow(pic)
                            ax_pic.axis('off')
                        fig.savefig(pos + '_avg.pdf')
                    #pdb.set_trace()
                else:
                    print("%s: Not more than one accepted spectrum available. Cannot do averaging" % pos)
        self.acf_average = np.array(acf_average)
        self.pos_average = np.array(pos_average)
