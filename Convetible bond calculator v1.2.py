import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit, QLabel,QTextBrowser
from PyQt5.QtCore import QCoreApplication
import ast
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import datetime

import easyquotation as eq
quote=eq.use('sina')

def getQuote(ticker):
	return float(list(quote.real(ticker).values())[0]['now'])

def getName(ticker):
	return list(quote.real(ticker).values())[0]['name']

def Outputdf(df,name='output'):
	df.to_csv(str(name)+'.csv')

def percent(float):
	return str('%.2f%%' % (float*100))

class Option:
	def __init__(self, type, s0, k, t, r, sigma,dv=0.0):
		if type=='c':
			self.type='Call'
			self.cp_sign=1.0
		elif type=='p':
			self.type='Put'
			self.cp_sign=-1.0

		self.s0=s0
		self.k=k
		self.t=t
		self.r=r
		self.sigma=sigma
		self.dv=dv
		self.d_1 = (np.log(self.s0 / self.k) + (self.r - self.dv + .5 * self.sigma ** 2) * self.t) / self.sigma / np.sqrt(self.t)
		self.d_2 = self.d_1 - self.sigma * np.sqrt(self.t)

	def price(self):
		return self.cp_sign * self.s0 * np.exp(-self.dv * self.t) * sps.norm.cdf(self.cp_sign * self.d_1) \
               - self.cp_sign * self.k * np.exp(-self.r * self.t) * sps.norm.cdf(self.cp_sign * self.d_2)

	def __repr__(self):
		string="\n*********************************\n"
		string+="Option type: "+str(self.type)+'\n'
		string+="Stock price: "+str(self.s0)+'\n'
		string+="Strike price: "+str(self.k)+'\n'
		string+="Maturity: "+str(self.t)+'\n'
		string+="Interest rate: "+str(self.r)+'\n'
		string+="Volatility: "+str(self.sigma)+'\n'
		string+="Dividend: "+str(self.dv)+'\n'
		string+="*********************************\n"
		string+=self.type+' price = '+str(self.price())+'\n'
		string+="*********************************\n"
		return string

class Bond:
	def __init__(self, coupon_list,t):
		self.coupon_list=coupon_list
		self.t=t

	def pv(self, r):
		length=len(self.coupon_list)
		pv=0.0
		for i, coupon in enumerate(self.coupon_list):
			if i>=length-round(self.t)-1:
				pv+=coupon*np.exp(-r*(self.t-(length-i-1)))
		return pv

	def fv(self, r):
		fv=self.pv(r)*np.exp(r*self.t)
		return fv

class ConvertibleBond:
	def __init__(self,coupon_list,t, convert_price, s0, r, sigma,dv=0.0, face_value=100, redemption_ratio=1.3):
		self.s0=s0
		self.redemption_ratio=redemption_ratio
		self.bond=Bond(coupon_list, t)
		self.t=t
		self.sigma=sigma
		self.k=convert_price*coupon_list[-1]/face_value
		self.r=r
		self.multiplier=face_value/convert_price
		self.option=Option("c", s0, self.k, t, r, sigma)

	def BondValue(self):
		return round(self.bond.pv(self.r),2)

	def OptionValue(self):
		return self.multiplier*self.option.price()

	def TotalValue(self):
		return round(self.BondValue()+self.OptionValue(),2)


	def ConvertValue(self,stock_price):
		return round(self.multiplier*stock_price,2)

	def TotalValue_ConvertValue_Ratio(self,stock_price):
		return (self.TotalValue()/self.ConvertValue(stock_price))

	def __repr__(self):
		string="\n*********************************\n"
		string+="Maturity: "+str(self.t)+'\n'
		string+="Interest rate: "+str(self.r)+'\n'
		string+="Volatility: "+str(self.sigma)+'\n'
		string+="*********************************\n"
		string+='Bond Value= '+str(self.BondValue())+'\n'
		string+='Option Value= '+str(self.OptionValue())+'\n'
		string+='Total Value= '+str(self.TotalValue())+'\n'
		string+="*********************************\n"
		return string



class CBPlot:

	def __init__(self,cb_ticker,stock_ticker, coupon_list, maturity_date, convert_price, interest_rate, volatility):
		self.cb_ticker=cb_ticker
		self.stock_ticker=stock_ticker
		self.cb_name=getName(cb_ticker)
		self.stock_name=getName(stock_ticker)
		self.cb_spot_price=getQuote(cb_ticker)
		self.stock_price=getQuote(stock_ticker)
		self.coupon_list=coupon_list
		self.maturity=((datetime.date(maturity_date[0],maturity_date[1],maturity_date[2])-datetime.date.today()).days)/365
		self.convert_price=convert_price
		self.interest_rate=interest_rate
		self.volatility=volatility
		self.cb=ConvertibleBond(self.coupon_list,self.maturity,self.convert_price,self.stock_price,self.interest_rate,self.volatility)

	def generateStockPriceSeries(self,lower=0.3,upper=2.0):
		simulated_stock_price=[]
		for i in range(100*round(self.convert_price*lower),100*round(self.convert_price*upper)):
			price=i/100
			simulated_stock_price.append(price)
		return simulated_stock_price

	def plot_cb(self):
		simulated_price_series=self.generateStockPriceSeries(upper=1.6)
		x_convert_value=[]
		y_cb_total_value=[]
		lower_bound=[]
		for simulated_price in simulated_price_series:
			x_convert_value.append(self.cb.ConvertValue(simulated_price))
			y_cb_total_value.append(ConvertibleBond(self.coupon_list,self.maturity,self.convert_price,simulated_price,self.interest_rate,self.volatility).TotalValue())
			lower_bound.append(max(ConvertibleBond(self.coupon_list,self.maturity,self.convert_price,simulated_price,self.interest_rate,self.volatility).BondValue(),self.cb.ConvertValue(simulated_price)))
		plt.figure(figsize=(7, 5))
		curve_1,=plt.plot(x_convert_value,y_cb_total_value,color='blue')
		curve_2,=plt.plot(x_convert_value,lower_bound, color='red')
		point_1 = plt.scatter(self.cb.ConvertValue(self.stock_price), self.cb_spot_price,c='green')
		plt.grid(ls='--')
		plt.title('Convertible Bond Analysis ('+str(self.cb_name)+')',fontproperties='SimHei',fontsize=20)
		plt.xlabel('Convertion Value')
		plt.ylabel('Price')
		plt.legend([curve_1,curve_2,point_1],['Black Scholes Value','Lower Bound',"(Conversion Value = "+str(self.cb.ConvertValue(self.stock_price))+',Spot Price = '+str(self.cb_spot_price)+')'],loc='upper left')
		plt.show()

	def df_cb(self):
		simulated_price_series=self.generateStockPriceSeries(upper=1.6)
		x_convert_value=[]
		y_cb_total_value=[]
		lower_bound=[]
		for simulated_price in simulated_price_series:
			x_convert_value.append(self.cb.ConvertValue(simulated_price))
			y_cb_total_value.append(ConvertibleBond(self.coupon_list,self.maturity,self.convert_price,simulated_price,self.interest_rate,self.volatility).TotalValue())
			lower_bound.append(max(ConvertibleBond(self.coupon_list,self.maturity,self.convert_price,simulated_price,self.interest_rate,self.volatility).BondValue(),self.cb.ConvertValue(simulated_price)))
		df=pd.DataFrame(index=simulated_price_series)
		df['Lower Bound']=lower_bound
		df['Black Scholes Value']=y_cb_total_value
		return df

	def __repr__(self):
		self.cb_spot_price=getQuote(self.cb_ticker)
		self.stock_price=getQuote(self.stock_ticker)
		self.cb=ConvertibleBond(self.coupon_list,self.maturity,self.convert_price,self.stock_price,self.interest_rate,self.volatility)
		string="********************************\n\n"
		string+='CB Name        = '+self.cb_name+'\n'
		string+='CB Spot Price  = '+str(self.cb_spot_price)+'\n\n'
		string+='Stock Name     = '+self.stock_name+'\n'
		string+='Stock Price    = '+str(self.stock_price)+'\n'
		string+="\n********************************\n\n"
		string+='Bond Floor Value     (债底价值) = '+str(self.cb.BondValue())+'\n'
		string+='Conversion Value     (转股价值) = '+str(self.cb.ConvertValue(self.stock_price))+'\n'
		string+='Black Scholes Value  (理论价值) = '+str(self.cb.TotalValue())+'\n'
		
		string+="\n********************************\n\n"
		string+="Above Bond Floor rate         (债底溢价率) = "+percent(self.cb_spot_price/self.cb.BondValue()-1)+'\n'
		string+="Conversion rate               (转股溢价率) = "+percent(self.cb_spot_price/self.cb.ConvertValue(self.stock_price)-1)+'\n'
		string+="Black Scholes Discount rate   (理论折价率) = "+percent(-(self.cb.TotalValue()/self.cb_spot_price-1))+'\n'
		string+="\n********************************\n\n"
		string+="Above Bond Floor rate （债底溢价率）\n= ((Spot Price / Bond Floor Value)-1)*100%\n"
		string+="\nConvertion rate (转股溢价率) \n= ((Spot Price / Conversion)-1)*100%\n\n"
		string+="Black Scholes Discount rate (理论折价率) \n= - ((Black Scholes Value / Spot Price)-1)*100%\n"
		string+="\n********************************"
		return string

class GUI(QWidget):
	def __init__(self):
		super().__init__()
		self.initialize_UI()	

	def initialize_UI(self):
		self.setGeometry(300, 300, 750, 600)
		self.setWindowTitle("Covertible Bond Calculator (by 13300200009@fudan.edu.cn)")
		
		self.label1=QLabel("CB_Ticker",self)
		self.label1.move(20,60)
		self.text1=QLineEdit("127003",self)
		self.text1.move(120,55)

		self.label2=QLabel("Stock_Ticker",self)
		self.label2.move(20,90)
		self.text2=QLineEdit("000861",self)
		self.text2.move(120,85)

		self.label3=QLabel("Coupon_List",self)
		self.label3.move(20,120)
		self.text3=QLineEdit("[0.5,0.7,1.0,1.5,1.8,110]",self)
		self.text3.move(120,115)
		self.text3.setFixedWidth(170)

		self.label4=QLabel("Maturity_Date",self)
		self.label4.move(20,150)
		self.text4=QLineEdit("[2022,6,8]",self)
		self.text4.move(120,145)

		self.label5=QLabel("Convert_Price",self)
		self.label5.move(20,180)
		self.text5=QLineEdit("3.03",self)
		self.text5.move(120,175)

		self.label6=QLabel("Interest_Rate",self)
		self.label6.move(20,210)
		self.text6=QLineEdit("0.05",self)
		self.text6.move(120,205)

		self.label7=QLabel("Volatility",self)
		self.label7.move(20,240)
		self.text7=QLineEdit("0.25",self)
		self.text7.move(120,235)
		
		self.label8=QLabel("1. Internet Connection Required.",self)
		self.label8.move(20,340)

		self.label9=QLabel("2. Copyright @ 13300200009@fudan.edu.cn \n   All rights reserved.",self)
		self.label9.move(20,370)

		self.label_output=QLabel('Output:',self)
		self.label_output.move(320,30)

		self.textbrowser=QTextBrowser(self)
		self.textbrowser.move(320,55)
		self.textbrowser.setFontPointSize(10)
		self.textbrowser.resize(400,500)

		self.button1=QPushButton('Calculate',self)
		self.button1.move(180,280)
		self.button1.clicked.connect(self.Calculate)

		

		self.show()

	def Calculate(self):
		self.cb_ticker=str(self.text1.text())
		self.stock_ticker=str(self.text2.text())
		self.coupon_list=ast.literal_eval(self.text3.text())
		self.maturity_date=ast.literal_eval(self.text4.text())
		self.convert_price=float(self.text5.text())
		self.interest_rate=float(self.text6.text())
		self.volatility=float(self.text7.text())
		self.conv_bond=CBPlot(self.cb_ticker,self.stock_ticker,self.coupon_list,self.maturity_date,self.convert_price,self.interest_rate,self.volatility)
		self.textbrowser.setText(str(self.conv_bond))
		self.conv_bond.plot_cb()
		Outputdf(self.conv_bond.df_cb())


app=QApplication(sys.argv)
a=GUI()
sys.exit(app.exec_())


