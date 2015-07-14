from __future__ import print_function
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
        return 55
    else:
        return 85

		
matplotlib.rcdefaults() 




#---------------------------------------------------------------------------------------------------------------------------------------------------				
#data load	
#---------------------------------------------------------------------------------------------------------------------------------------------------		

noReg=pd.read_csv('noRegression0713.csv', keep_default_na=False, na_values=["NA"],dtype={'Carry+Roll)/Std':np.float64})
noReg[["(Carry+Roll)/Std","Carry(3m)","Roll(3m)"]]=noReg[["(Carry+Roll)/Std","Carry(3m)","Roll(3m)"]].convert_objects(convert_numeric=True)
noReg['Tenor']=noReg.apply(lambda x: x['PT_Tenor1']+x['PT_Tenor2']+x['PT_Tenor3'] ,axis=1)
#noReg['bubblesize']=noReg.apply(lambda x: 400*abs(x['(Carry+Roll)/Std']),axis=1)
noReg['bubblesize']=noReg.apply(lambda x: abs(x['(Carry+Roll)/Std']),axis=1)
noReg['bubblecolor']=noReg.apply(lambda x: mapcolor(x),axis=1)
noReg['quad']=noReg.apply(quad,axis=1)

Reg=pd.read_csv('Regression0713.csv', keep_default_na=False, na_values=["NA"])
CCy=['USD', 'CAD','GBP','EUR','JPY']
itype=['switch','fly']
scores=['C','R','P']

#---------------------------------------------------------------------------------------------------------------------------------------------------		
#Top trade by CCY
#---------------------------------------------------------------------------------------------------------------------------------------------------		

for iccy in CCy:
	topTrades = noReg[(noReg['weighted/unweighted']=='unweighted') & (noReg['PT_Currency']==iccy) & (noReg['Main Product Type']=='swap') & (noReg['PT_Priority']=="p1") & (noReg['PT Name'].isin( ['switch', 'fly', 'outright']))]
	topTrades1 =topTrades.sort('bubblesize', ascending=False).groupby('quad')
	topTradesFinal=topTrades1.get_group(topTrades1.groups.keys()[0]).head(5)
	for i in range(1,len(topTrades1.groups.keys())):
		topTradesFinal=topTradesFinal.append(topTrades1.get_group(topTrades1.groups.keys()[i]).head(5))
		
	maxsize=np.max(topTradesFinal['bubblesize'])
	minsize=np.min(topTradesFinal['bubblesize'])	
	avgsize=np.mean(topTradesFinal['bubblesize'])
	diff=maxsize-minsize
	topTradesFinal['bubblesizeNormal']=topTradesFinal.apply(lambda x: ((x['bubblesize']-minsize)/diff)*5000,axis=1)
	#topTradesFinal['bubblesizeNormal']=topTradesFinal.apply(lambda x: (abs(x['bubblesize']-avgsize)/diff)*5000,axis=1)	
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
	#axes2.xaxis.set_ticks_position('bottom')
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())	
	#axes2.yaxis.set_ticks_position('left')	
	axes2.spines['bottom'].set_position(('data',0)) 
	axes2.spines['left'].set_position(('data',0))  
	axes2.scatter(topTradesFinal['Corr_PC1'],topTradesFinal['Corr_PC2'],s=topTradesFinal['bubblesizeNormal'],marker='o',c=topTradesFinal['bubblecolor'],alpha=0.55)

	for aa,x,y in zip(topTradesFinal['Tenor'],topTradesFinal['Corr_PC1'],topTradesFinal['Corr_PC2']):
		axes2.annotate(aa,xy=(x,y),fontsize=11)


	filename="images/Top" + topTradesFinal.iloc[0]['PT_Currency'] + "Trades.svg"
	fig.savefig(filename,dpi=500)
	plt.close()


	
#---------------------------------------------------------------------------------------------------------------------------------------------------			
#Picking top PT1 trades from regression table
#---------------------------------------------------------------------------------------------------------------------------------------------------		

topTrades = Reg[(Reg['PT_Priority(dependent var)']=="p1") & (Reg['RSq']>=0.8) &  (Reg['single/multi currency']=="single ccy") & (Reg['Main Product Type'] == 'swap')]
topTrades['Zabs']=topTrades.apply(lambda x: abs(x['Z(5b)']),axis=1)
for iccy in CCy:
	topTrades_gpd=topTrades[topTrades['PT_Currency(dependent var)']==iccy].sort('Zabs', ascending=False).groupby('PT(dependent var)')
	topTrades_Z=topTrades_gpd.get_group(topTrades_gpd.groups.keys()[0]).head(2)
	for i in range(1,len(topTrades_gpd.groups.keys())):
		topTrades_Z=topTrades_Z.append(topTrades_gpd.get_group(topTrades_gpd.groups.keys()[i]).head(2))
	
		
	top10=topTrades_Z.sort('Zabs',ascending=False).head(n=10)
	top10print={'dep':top10['PT(dependent var)'], 'indep':top10['PT(independent var)'], 'Z':np.round(top10['Z(5b)'],2), 'P':np.round(map(float,top10['(Carry+Roll)/Std']),2)}
	top10print1=pd.DataFrame(top10print,columns=['dep','indep','Z','P'])
	#print iccy + "\n"
	#print top10print1.to_latex(index=False,longtable=False)
	log=open("images/" + iccy + "_regression.txt","w")
	print (top10print1.to_latex(index=False,longtable=False), file=log)
	log.close()


#---------------------------------------------------------------------------------------------------------------------------------------------------		
#Top switches and Top Flies 
#---------------------------------------------------------------------------------------------------------------------------------------------------		

for iccy in CCy:
	for jtype in itype:
		outrightsUSD = noReg[(noReg['weighted/unweighted']=='unweighted') & (noReg['PT_Currency']==iccy) & (noReg['PT Name']==jtype) & (noReg['Main Product Type']=='swap')]
		abc=outrightsUSD.sort('bubblesize', ascending=False ).groupby('quad')
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
		#axes2.xaxis.set_ticks_position('bottom')
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		axes2.spines['bottom'].set_position(('data',0)) 
		#axes2.yaxis.set_ticks_position('left')
		axes2.yaxis.set_ticks_position('none')
		axes2.spines['left'].set_position(('data',0))  
		axes2.scatter(ff1['Corr_PC1'],ff1['Corr_PC2'],s=ff1['bubblesizeNormal'],marker='o',c=ff1['bubblecolor'],alpha=0.75)

		for aa,x,y in zip(ff1['Tenor'],ff1['Corr_PC1'],ff1['Corr_PC2']):
			axes2.annotate(aa,xy=(x,y),fontsize=10)

		filename="images/" + ff1.iloc[0]['PT Name']+ ff1.iloc[0]['PT_Currency'] + ".svg"
		fig.savefig(filename,dpi=500)
		plt.close()


		
		
#---------------------------------------------------------------------------------------------------------------------------------------------------				
#Generating weighted switches and flies tablefor jtype in itype:
#---------------------------------------------------------------------------------------------------------------------------------------------------		

for jtype in itype:
	ff1 = noReg[(noReg['PT Name']==jtype) & (noReg['PT_Priority']=="p1") & (noReg['Main Product Type']=='swap') & (noReg['weighted/unweighted']=='weighted')]
	ff1['Weights'] = ff1.apply(lambda x:'[' + str(x['PT_Coeff1']) +',' + str(x['PT_Coeff2']) + ',' + str(x['PT_Coeff3']) + ']',axis=1)
	ff1['Tenor1']=ff1.apply(lambda x: float(x['PT_Tenor1'][:-1]),axis=1)
	ff1['Tenor2']=ff1.apply(lambda x: float(x['PT_Tenor2'][:-1]),axis=1)
	ff1['C']=ff1.apply(lambda x:x['Carry(3m)']/x['Std'],axis=1)
	ff1['R']=ff1.apply(lambda x:x['Roll(3m)']/x['Std'],axis=1)
	ff1['P']=ff1.apply(lambda x:(x['Carry(3m)']+x['Roll(3m)'])/x['Std'],axis=1)
	if (jtype=='switch'):
		ff2=ff1.groupby('PT_Currency')
		ff3=pd.DataFrame()
		for i in range(0,len(ff2.groups.keys())):
			ccy=ff2.groups.keys()[i]
			ff3_tmp=ff2.get_group(ccy)
			ff3_tmp=ff3_tmp.sort(['Tenor1','Tenor2'])
			for iscore in scores:
				avg1=ff3_tmp[iscore].mean()
				max1=ff3_tmp[iscore].max()
				min1=ff3_tmp[iscore].min()				
				ff3_tmp[iscore + "_score"]=ff3_tmp.apply(lambda x: ((-x[iscore]-avg1)/(min1-avg1))*10 if x[iscore] < avg1 else ((x[iscore]-avg1)/(max1-avg1))*10,axis=1)
			ff3=ff3.append(ff3_tmp)			
	elif (jtype=='fly'):
		ff1['Tenor3']=ff1.apply(lambda x: float(x['PT_Tenor3'][:-1]),axis=1)
		ff2=ff1.groupby('PT_Currency')
		ff4=pd.DataFrame()
		for i in range(0,len(ff2.groups.keys())):
			ccy=ff2.groups.keys()[i]
			ff4_tmp=ff2.get_group(ccy)
			ff4_tmp=ff4_tmp.sort(['Tenor2','Tenor1','Tenor3'])
			for iscore in scores:
				avg1=ff4_tmp[iscore].mean()
				max1=ff4_tmp[iscore].max()
				min1=ff4_tmp[iscore].min()				
				ff4_tmp[iscore + "_score"]=ff4_tmp.apply(lambda x: ((-x[iscore]-avg1)/(min1-avg1))*10 if x[iscore] < avg1 else ((x[iscore]-avg1)/(max1-avg1))*10,axis=1)
			ff4=ff4.append(ff4_tmp)		
	

zz_switch={'CCY':ff3['PT_Currency'],'Switch':ff3['Tenor'],'Weights':ff3['Weights'],'Carry':np.round(ff3['Carry(3m)'],2),'Roll':np.round(ff3['Roll(3m)'],2),'Vol':np.round(ff3['Std'],2),'R':np.round(ff3['R_score'],1),'C':np.round(ff3['C_score'],1),'P':np.round(ff3['P_score'],1)}	
zz_fly={'CCY':ff4['PT_Currency'],'Flies':ff4['Tenor'],'Weights':ff4['Weights'],'Carry':np.round(ff4['Carry(3m)'],2),'Roll':np.round(ff4['Roll(3m)'],2),'Vol':np.round(ff4['Std'],2),'R':np.round(ff4['R_score'],1),'C':np.round(ff4['C_score'],1),'P':np.round(ff4['P_score'],1)}
zz1_switch=pd.DataFrame(zz_switch, columns=['CCY','Switch','Weights','Carry','Roll','Vol','R','C','P'])
zz1_fly=pd.DataFrame(zz_fly, columns=['CCY','Flies','Weights','Carry','Roll','Vol','R','C','P'])

log = open("images/Weightedswitch.txt", "w")
print (zz1_switch.to_latex(index=False,longtable=False), file=log)
log.close()

log = open("images/Weightedfly.txt", "w")
print (zz1_fly.to_latex(index=False,longtable=False), file=log)
log.close()		
		
		
#---------------------------------------------------------------------------------------------------------------------------------------------------		
#picking the top switches and flies from PT1 trades
#---------------------------------------------------------------------------------------------------------------------------------------------------

for iccy in CCy:
	for jtype in itype:
		ff1 = noReg[(noReg['PT_Currency']==iccy) & (noReg['PT Name']==jtype) & (noReg['PT_Priority']=="p1") & (noReg['Main Product Type']=='swap') & (noReg['weighted/unweighted']=='unweighted')]
		ff1['Tenor1']=ff1.apply(lambda x: float(x['PT_Tenor1'][:-1]),axis=1)
		ff1['Tenor2']=ff1.apply(lambda x: float(x['PT_Tenor2'][:-1]),axis=1)
		if (jtype=='switch'):
			ff1=ff1.sort(['Tenor1','Tenor2'])
		elif (jtype=='fly'):
			ff1['Tenor3']=ff1.apply(lambda x: float(x['PT_Tenor3'][:-1]),axis=1)
			ff1=ff1.sort(['Tenor2','Tenor1', 'Tenor3'])
			
		#outrightsUSD = noReg[(noReg['PT_Currency']==iccy) & (noReg['PT Name']==jtype) & (noReg['PT_Priority']=="p1") & (noReg['Main Product Type']=='swap')]
		#abc=outrightsUSD.sort('bubblesize').groupby('quad')
		#ff1=abc.get_group(abc.groups.keys()[0]).head(4)
		#for i in range(1,len(abc.groups.keys())):
		#	ff1=ff1.append(abc.get_group(abc.groups.keys()[i]).head(4))

		zz={ff1.iloc[0]['PT Name']:ff1['Tenor'],'Lvl':np.round(ff1['Level_Current'],2), 'Lvl-L':np.round(ff1['Level_Low(3m)'],2),'Lvl-H':np.round(ff1['Level_High(3m)'],2),'Carry':np.round(ff1['Carry(3m)'],2),'Roll':np.round(ff1['Roll(3m)'],2),'Vol':np.round(ff1['Std'],2),'Z':np.round(ff1['Z'],2),'P':np.round(ff1['(Carry+Roll)/Std'],2),'Duration':ff1['Type_PC1'],'Curve':ff1['Type_PC2']}
		zz1=pd.DataFrame(zz,columns=[ff1.iloc[0]['PT Name'],'Lvl','Lvl-L','Lvl-H','Carry','Roll','Vol','Z','P','Duration', 'Curve'])
		#print iccy + " " + jtype + "\n"
		#print zz1.head(n=25).to_latex(index=False,longtable=False)
		log=open("images/" + iccy + "-" + jtype + ".txt","w")
		print (zz1.to_latex(index=False,longtable=False), file=log)
		log.close()
		


# (Outrights bubble and stacked bar plots) fig1 and fig2
#---------------------------------------------------------------

for iccy in CCy:
	outrightsUSD = noReg[(noReg['PT_Currency']==iccy) & (noReg['PT Name']=='outright') & (noReg['Main Product Type']=='swap')]
	#Z=outrightsUSD.apply(lambda x: -x['Z'],axis=1)
	Z=map(float,outrightsUSD['Z'])
	tenor = outrightsUSD.apply(lambda x: float(x['Tenor'][:-1]),axis=1)
	temp = dict(zip(tenor, outrightsUSD['(Carry+Roll)/Std']))
	levels = outrightsUSD.apply(lambda x:'(C:' + str(np.round(x['Level_Current'],0)) + ',L:' + str(np.round(x['Level_Low(3m)'],0)) + ',H:' + str(np.round(x['Level_High(3m)'],0)) + ')',axis=1)
	temp1=dict(zip(outrightsUSD['Z_SVD'], Z))
	labels =zip(outrightsUSD['Z_SVD'], levels, outrightsUSD['Tenor'])	
	filtered_temp1 = {k:v for (k,v) in temp1.items() if k != ""}
	labels_lvl = {k:v for (k,v,w) in labels if k != ""}
	labels_tenor = {k:w for (k,v,w) in labels if k != ""}

	latexify()
	#fig, axes = plt.subplots(figsize=(12, 6))
	fig=plt.figure()
	axes=fig.add_axes([0,0,1,1])
	axes.set_title("OutRights(rich/cheap)")
	axes.set_xlim([min(map(float,filtered_temp1.keys()))-0.5,max(map(float,filtered_temp1.keys()))+0.5])
	axes.set_ylim([min(filtered_temp1.values())-0.5,max(filtered_temp1.values())+0.5])
	#axes.spines['bottom'].set_position(('data',min(0,max(filtered_temp1.values())+0.5)))
	u=lambda ymin,ymax: ymax if ymax<0 else ymin if ymin>0  else 0 
	axes.spines['bottom'].set_position(('data',u(min(filtered_temp1.values())-0.5,max(filtered_temp1.values())+0.5)))
	axes.spines['left'].set_position(('data',0))
	axes.spines['right'].set_color('none')
	axes.spines['top'].set_color('none')
	#axes.xaxis.set_ticks_position('bottom')
	#axes.yaxis.set_ticks_position('left')
	axes.xaxis.set_ticks_position('none')
	axes.yaxis.set_ticks_position('none')

	for spine in ['left', 'bottom']:    
		axes.spines[spine].set_linewidth(0.5)

	axes.plot(map(float,filtered_temp1.keys()), filtered_temp1.values(), color="blue", lw=2, ls='*', marker='d', markersize=5)
	axes.set_xlabel('Z SVD')
	axes.set_ylabel('Z')
	y_min=axes.get_ylim()[0]
	y_max=axes.get_ylim()[1]
	sig=y_max-y_min
	for bb,aa,x,y in zip(labels_tenor.values(), labels_lvl.values(),filtered_temp1.keys(),filtered_temp1.values()):
		axes.annotate(aa,xy=(x,y), color='blue', alpha=0.8)
		axes.annotate(bb,xy=(float(x),float(y)+0.2*(abs(float(y)-y_min)/sig)), color='blue', alpha=0.8)

	#fig.tight_layout()

	filename="images/Outrights" + outrightsUSD.iloc[0]['PT_Currency'] + "Trades_scatter.svg"
	fig.savefig(filename,dpi=500)
	plt.close()



	#fig, axes = plt.subplots(figsize=(12, 6))
	fig=plt.figure()
	axes=fig.add_axes([0,0,1,1])
	axes.set_title("OutRights(carry/roll)")
	axes.set_xlim([0,30.0])
	axes.set_ylim([0,max(outrightsUSD['Carry(3m)'] + outrightsUSD['Roll(3m)'])+5.0])
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
	plt.close()
	
