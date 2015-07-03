import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

pd.set_option('html', False)
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 50)


from math import sqrt
import matplotlib

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'font.size': 7, # was 10              
              'axes.labelsize': 7, # fontsize for x and y labels (was 10)
              'axes.titlesize': 7,              
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)






def quad(elem):
    if (elem['Corr_PC1']>0 and elem['Corr_PC2']>0):
        return 'Q1'
    elif (elem['Corr_PC1']>0 and elem['Corr_PC2']<0):
        return 'Q4'
    elif (elem['Corr_PC1']<0 and elem['Corr_PC2']>0):
        return 'Q2'
    elif (elem['Corr_PC1']<0 and elem['Corr_PC2']<0):
        return 'Q3'

def mapcolor(elem):
    temp=math.copysign(1,elem['Carry(3m)'])
    if temp==1:
        return 'g'
    else:
        return 'r'

		

#data load	
#-----------------------------------------------------------------------------------	
noReg=pd.read_csv('noRegression.csv', keep_default_na=False, na_values=["NA"],dtype={'Carry+Roll)/Std':np.float64})
noReg[["(Carry+Roll)/Std","Carry(3m)","Roll(3m)"]]=noReg[["(Carry+Roll)/Std","Carry(3m)","Roll(3m)"]].convert_objects(convert_numeric=True)
noReg['Tenor']=noReg.apply(lambda x: x['PT_Tenor1']+x['PT_Tenor2']+x['PT_Tenor3'] ,axis=1)
#noReg['bubblesize']=noReg.apply(lambda x: 400*abs(x['(Carry+Roll)/Std']),axis=1)
noReg['bubblesize']=noReg.apply(lambda x: abs(x['(Carry+Roll)/Std']),axis=1)
noReg['bubblecolor']=noReg.apply(lambda x: mapcolor(x),axis=1)
noReg['quad']=noReg.apply(quad,axis=1)



#Top trade by CCY
#-------------------------------------------------------------------------------------
topTrades = noReg[(noReg['PT_Currency']=="USD") & (noReg['Main Product Type']=='swap')]
topTrades1 =topTrades.sort('bubblesize').groupby('quad')
topTradesFinal=topTrades1.get_group(topTrades1.groups.keys()[0]).head(4)
for i in range(1,len(topTrades1.groups.keys())):
    topTradesFinal=topTradesFinal.append(topTrades1.get_group(topTrades1.groups.keys()[i]).head(4))
	
maxsize=max(topTradesFinal['bubblesize'])
minsize=min(topTradesFinal['bubblesize'])	
diff=maxsize-minsize
topTradesFinal['bubblesizeNormal']=topTradesFinal.apply(lambda x: ((x['bubblesize']-minsize)/diff)*5000,axis=1)
	
fig=plt.figure()
axes1 =fig.add_axes([0,0,1,1])
axes2 = fig.add_axes([0.08,0.1,0.85,0.85], frameon=True,alpha=0)
axes1.set_xlim(xmin=-1, xmax=1)
axes1.set_ylim(ymin=-1, ymax=1)
axes1.annotate('Bull',(-0.5,-0.9))
axes1.annotate('Bear',(0.5,-0.9))
axes1.annotate('Bull',(-0.5,0.92))
axes1.annotate('Bear',(0.5,0.92))
axes1.annotate('Steepener',(-0.95,0.5),rotation=90)
axes1.annotate('Flattener',(-0.95,-0.5),rotation=90)
axes1.annotate('Steepener',(0.90,0.5),rotation=270)
axes1.annotate('Flattener',(0.90,-0.5),rotation=270)
axes1.axis('off')
axes2.set_xlim(xmin=-1.1, xmax=1.1)
axes2.set_ylim(ymin=-1.1, ymax=1.1)
axes2.spines['right'].set_color('none')
axes2.spines['top'].set_color('none')
axes2.xaxis.set_ticks_position('bottom')
axes2.spines['bottom'].set_position(('data',0)) 
axes2.yaxis.set_ticks_position('left')
axes2.spines['left'].set_position(('data',0))  
axes2.scatter(topTradesFinal['Corr_PC1'],topTradesFinal['Corr_PC2'],s=topTradesFinal['bubblesizeNormal'],marker='o',c=topTradesFinal['bubblecolor'],alpha=0.75)

for aa,x,y in zip(topTradesFinal['Tenor'],topTradesFinal['Corr_PC1'],topTradesFinal['Corr_PC2']):
    axes2.annotate(aa,xy=(x,y),fontsize=10)


filename="images/Top" + topTradesFinal.iloc[0]['PT_Currency'] + "Trades.svg"
fig.savefig(filename,dpi=500)


#Picking top PT1 trades
CCy=['USD', 'CAD','GBP','EUR','JPY']
for i in CCy:
	topTrades = noReg[(noReg['PT_Currency']==i) & (noReg['PT_Priority']=="p1") &(noReg['Main Product Type']=='swap')]
	topTrades1 =topTrades.sort('bubblesize').groupby('quad')
	topTradesFinal=topTrades1.get_group(topTrades1.groups.keys()[0]).head(4)
	for i in range(1,len(topTrades1.groups.keys())):
		topTradesFinal=topTradesFinal.append(topTrades1.get_group(topTrades1.groups.keys()[i]).head(4))

	zz={'topTrades':topTradesFinal['Tenor'],'Lvl':np.round(topTradesFinal['Level_Current'],2), 'Lvl Low':np.round(topTradesFinal['Level_Low(3m)'],2),'Lvl High':np.round(topTradesFinal['Level_High(3m)'],2),'Carry':np.round(topTradesFinal['Carry(3m)'],2),'Roll':np.round(topTradesFinal['Roll(3m)'],2),'DailyVol':np.round(topTradesFinal['Std'],2),'Z PCA':np.round(topTradesFinal['Z'],2),'p-score':np.round(topTradesFinal['(Carry+Roll)/Std'],2),'Duration':topTradesFinal['Type_PC1'],'Curve':topTradesFinal['Type_PC2\
	']}
	zz1=pd.DataFrame(zz,columns=['topTrades','Lvl','Lvl Low','Lvl High','Carry','Roll','DailyVol','Z PCA','p-score','Duration', 'Curve'])
	print zz1.to_latex(index=False)




#Top switches and Top Flies 
#--------------------------------------------------------------------------
outrightsUSD = noReg[(noReg['PT_Currency']=="USD") & (noReg['PT Name']=='switch') & (noReg['Main Product Type']=='swap')]
abc=outrightsUSD.sort('bubblesize').groupby('quad')
ff1=abc.get_group(abc.groups.keys()[0]).head(4)
for i in range(1,len(abc.groups.keys())):
    ff1=ff1.append(abc.get_group(abc.groups.keys()[i]).head(4))

	
maxsize1=max(ff1['bubblesize'])
minsize1=min(ff1['bubblesize'])	
diff1=maxsize1-minsize1
ff1['bubblesizeNormal']=ff1.apply(lambda x: ((x['bubblesize']-minsize1)/diff1)*5000,axis=1)
	
fig=plt.figure()
axes1 =fig.add_axes([0,0,1,1])
axes2 = fig.add_axes([0.08,0.1,0.85,0.85], frameon=True,alpha=0)
axes1.set_xlim(xmin=-1, xmax=1)
axes1.set_ylim(ymin=-1, ymax=1)
axes1.annotate('Bull',(-0.5,-0.9))
axes1.annotate('Bear',(0.5,-0.9))
axes1.annotate('Bull',(-0.5,0.92))
axes1.annotate('Bear',(0.5,0.92))
axes1.annotate('Steepener',(-0.95,0.5),rotation=90)
axes1.annotate('Flattener',(-0.95,-0.5),rotation=90)
axes1.annotate('Steepener',(0.90,0.5),rotation=270)
axes1.annotate('Flattener',(0.90,-0.5),rotation=270)
axes1.axis('off')
axes2.set_xlim(xmin=-1.1, xmax=1.1)
axes2.set_ylim(ymin=-1.1, ymax=1.1)
axes2.spines['right'].set_color('none')
axes2.spines['top'].set_color('none')
axes2.xaxis.set_ticks_position('bottom')
axes2.spines['bottom'].set_position(('data',0)) 
axes2.yaxis.set_ticks_position('left')
axes2.spines['left'].set_position(('data',0))  
axes2.scatter(ff1['Corr_PC1'],ff1['Corr_PC2'],s=ff1['bubblesizeNormal'],marker='o',c=ff1['bubblecolor'],alpha=0.75)

for aa,x,y in zip(ff1['Tenor'],ff1['Corr_PC1'],ff1['Corr_PC2']):
    axes2.annotate(aa,xy=(x,y),fontsize=10)

filename="images/" + ff1.iloc[0]['PT Name']+ ff1.iloc[0]['PT_Currency'] + ".svg"
fig.savefig(filename,dpi=500)


#picking the top switches and flies from PT1 trades

CCy=['USD', 'CAD','GBP','EUR','JPY']
itype=['switch','fly']
for iccy in CCy:
	for jtype in itype:
		outrightsUSD = noReg[(noReg['PT_Currency']==iccy) & (noReg['PT Name']==jtype) & (noReg['PT_Priority']=="p1") & (noReg['Main Product Type']=='swap')]
		abc=outrightsUSD.sort('bubblesize').groupby('quad')
		ff1=abc.get_group(abc.groups.keys()[0]).head(4)
		for i in range(1,len(abc.groups.keys())):
			ff1=ff1.append(abc.get_group(abc.groups.keys()[i]).head(4))

		zz={ff1.iloc[0]['PT Name']:ff1['Tenor'],'Lvl':np.round(ff1['Level_Current'],2), 'Lvl Low':np.round(ff1['Level_Low(3m)'],2),'Lvl High':np.round(ff1['Level_High(3m)'],2),'Carry':np.round(ff1['Carry(3m)'],2),'Roll':np.round(ff1['Roll(3m)'],2),'DailyVol':np.round(ff1['Std'],2),'Z PCA':np.round(ff1['Z'],2),'p-score':np.round(ff1['(Carry+Roll)/Std'],2),'Duration':ff1['Type\_PC1'],'Curve':ff1['Type\_PC2\
		']}
		zz1=pd.DataFrame(zz,columns=[ff1.iloc[0]['PT Name'],'Lvl','Lvl Low','Lvl High','Carry','Roll','DailyVol','Z PCA','p-score','Duration', 'Curve'])
		print iccy + " " + jtype + "\n"
		print zz1.to_latex(index=False)



# (Outrights bubble and stacked bar plots) fig1 and fig2
#---------------------------------------------------------------
outrightsUSD = noReg[(noReg['PT_Currency']=="USD") & (noReg['PT Name']=='outright') & (noReg['Main Product Type']=='swap')]
Z=outrightsUSD.apply(lambda x: -x['Z'],axis=1)
tenor = outrightsUSD.apply(lambda x: float(x['Tenor'][:-1]),axis=1)
temp = dict(zip(tenor, outrightsUSD['(Carry+Roll)/Std']))
temp1=dict(zip(outrightsUSD['Z_SVD'], Z))
filtered_temp1 = {k:v for (k,v) in temp1.items() if k != ""}
levels = outrightsUSD.apply(lambda x:'C:' + str(np.round(x['Level_Current'],2)) + ',L:' + str(np.round(x['Level_Low(3m)'],2)) + ',H:' + str(np.round(x['Level_High(3m)'],2)),axis=1)



latexify()
#fig, axes = plt.subplots(figsize=(12, 6))
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.set_title("OutRights(rich/cheap)")
axes.set_xlim([-3.0,4.0])
axes.set_ylim([0,4.0])
axes.spines['bottom'].set_position(('data',0))
axes.spines['left'].set_position(('data',0))
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
#axes.xaxis.set_ticks_position('bottom')
#axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('none')
axes.yaxis.set_ticks_position('none')

for spine in ['left', 'bottom']:    
    axes.spines[spine].set_linewidth(0.5)

axes.plot(filtered_temp1.keys(), filtered_temp1.values(), color="blue", lw=2, ls='*', marker='s')
axes.set_xlabel('Z SVD')
axes.set_ylabel('Z')
for aa,x,y in zip(levels,filtered_temp1.keys(),filtered_temp1.values()):
    axes.annotate(aa,xy=(x,y), color='blue', alpha=0.5)

#fig.tight_layout()

filename="images/Outrights" + outrightsUSD.iloc[0]['PT_Currency'] + "Trades_scatter.svg"
fig.savefig(filename,dpi=500)



#fig, axes = plt.subplots(figsize=(12, 6))
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.set_title("OutRights(carry/roll)")
axes.set_xlim([0,30.0])
axes.spines['bottom'].set_position(('data',0))
axes.spines['left'].set_position(('data',0))
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.xaxis.set_ticks_position('bottom')
axes.yaxis.set_ticks_position('left')
axes.bar(tenor.astype(float),outrightsUSD['Carry(3m)'],width=0.8,color='b',alpha=0.9,edgecolor='none',linewidth=0, label="carry(left)")
axes.bar(tenor.astype(float),outrightsUSD['Roll(3m)'],width=0.8,color='b', bottom=outrightsUSD['Carry(3m)'], alpha=0.4,edgecolor='none',linewidth=0, label="roll(left)")
ax2=axes.twinx()
ax2.spines['top'].set_color('none')
ax2.set_xlim([0,32.0])
for spine in ['left', 'bottom','right']:    
    ax2.spines[spine].set_linewidth(0.3)
    axes.spines[spine].set_linewidth(0.3)

ax2.plot(temp.keys(),temp.values(),lw=1.1,color="green",alpha=0.5,label="risk adjust carry+roll" )
#ax2.set_ylabel("risk adj carry+roll",labelpad=0.2)

h1, l1 = axes.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
axes.legend(h1+h2, l1+l2, loc=1)
filename="images/Outrights" + outrightsUSD.iloc[0]['PT_Currency'] + "Trades_stackedBar.svg"
fig.savefig(filename,dpi=500)
