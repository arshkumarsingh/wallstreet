import requests
import xml.etree.ElementTree as ET

from scipy.interpolate import interp1d
from numpy import sqrt, log, exp
from scipy.stats import norm
from scipy.optimize import fsolve

from wallstreet.constants import *

import xml.etree.ElementTree as ET

def riskfree():
    """
    Function to retrieve the current risk-free interest rate from the US Treasury
    website. If the request fails, it returns a lambda function that returns a
    default risk-free rate.

    Returns:
        A function that takes a float x and returns the risk-free interest rate
        for that x.
    """
    try:
        # Send a GET request to the US Treasury website
        r = requests.get(TREASURY_URL)

        # Parse the XML response
        root = ET.fromstring(r.text)
        days = root.findall('.//G_BC_CAT')
        last = days[-1]

        def parse(node):
            """
            Helper function to parse a node and convert it to a float.

            Args:
                node: An XML node to parse.

            Returns:
                A float representation of the node's text.
            """
            return float(node.text)

        # Parse various nodes in the XML response and convert them to floats
        m1 = parse(last.find('BC_1MONTH'))
        m2 = parse(last.find('BC_2MONTH'))
        m3 = parse(last.find('BC_3MONTH'))
        m6 = parse(last.find('BC_6MONTH'))
        y1 = parse(last.find('BC_1YEAR'))
        y2 = parse(last.find('BC_2YEAR'))
        y3 = parse(last.find('BC_3YEAR'))
        y5 = parse(last.find('BC_5YEAR'))
        y7 = parse(last.find('BC_7YEAR'))
        y10 = parse(last.find('BC_10YEAR'))
        y20 = parse(last.find('BC_20YEAR'))
        y30 = parse(last.find('BC_30YEAR'))

        # Define the years and their corresponding interest rates
        years = (0, 1/12, 2/12, 3/12, 6/12, 12/12, 24/12, 36/12, 60/12, 84/12, 120/12, 240/12, 360/12)
        rates = (OVERNIGHT_RATE, m1/100, m2/100, m3/100, m6/100, y1/100, y2/100, y3/100, y5/100, y7/100, y10/100, y20/100, y30/100)

        # Use scipy's interp1d function to create an interpolating function that
        # takes a float x and returns the risk-free interest rate for that x.
        return interp1d(years, rates)
    except Exception:
        # If the request fails, return a lambda function that returns a default
        # risk-free rate.
        return lambda x: FALLBACK_RISK_FREE_RATE

class BlackandScholes:
    """
    Class implementing the Black-Scholes model for option pricing.
    """

    def __init__(self, S, K, T, price, r, option, q=0):
        """
        Initialize the BlackandScholes class with stock price S, strike price K,
        time to maturity T, option price price, risk-free rate r, option type
        option, and dividend yield q.
        """
        self.S, self.K, self.T, self.option, self.q = S, K, T, option, q
        self.r = r
        self.opt_price = price
        self.impvol = self.implied_volatility()

    @staticmethod
    def _BlackScholesCall(S, K, T, sigma, r, q):
        """
        Calculate the price of a call option using the Black-Scholes model.
        """
        d1 = (log(S/K) + (r - q + (sigma**2)/2)*T)/(sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

    @staticmethod
    def _BlackScholesPut(S, K, T, sigma, r, q):
        """
        Calculate the price of a put option using the Black-Scholes model.
        """
        d1 = (log(S/K) + (r - q + (sigma**2)/2)*T)/(sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return  K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

    def _fprime(self, sigma):
        """
        Calculate the derivative of the option price with respect to the
        implied volatility.
        """
        logSoverK = log(self.S/self.K)
        n12 = ((self.r + sigma**2/2)*self.T)
        numerd1 = logSoverK + n12
        d1 = numerd1/(sigma*sqrt(self.T))
        return self.S*sqrt(self.T)*norm.pdf(d1)*exp(-self.r*self.T)

    def BS(self, S, K, T, sigma, r, q):
        """
        Calculate the option price using the Black-Scholes model.
        """
        if self.option == 'Call':
            return self._BlackScholesCall(S, K, T, sigma, r, q)
        elif self.option == 'Put':
            return self._BlackScholesPut(S, K, T, sigma, r, q)

    def implied_volatility(self):
        """
        Calculate the implied volatility of the option.
        """
        impvol = lambda x: self.BS(self.S, self.K, self.T, x, self.r, self.q) - self.opt_price
        iv = fsolve(impvol, SOLVER_STARTING_VALUE, fprime=self._fprime, xtol=IMPLIED_VOLATILITY_TOLERANCE)
        return iv[0]

    def delta(self):
        """
        Calculate the delta of the option.
        """
        h = DELTA_DIFFERENTIAL
        p1 = self.BS(self.S + h, self.K, self.T, self.impvol, self.r, self.q)
        p2 = self.BS(self.S - h, self.K, self.T, self.impvol, self.r, self.q)
        return (p1-p2)/(2*h)

    def gamma(self):
        """
        Calculate the gamma of the option.
        """
        h = GAMMA_DIFFERENTIAL
        p1 = self.BS(self.S + h, self.K, self.T, self.impvol, self.r, self.q)
        p2 = self.BS(self.S, self.K, self.T, self.impvol, self.r, self.q)
        p3 = self.BS(self.S - h, self.K, self.T, self.impvol, self.r, self.q)
        return (p1 - 2*p2 + p3)/(h**2)

    def vega(self):
        """
        Calculate the vega of the option.
        """
        h = VEGA_DIFFERENTIAL
        p1 = self.BS(self.S, self.K, self.T, self.impvol + h, self.r, self.q)
        p2 = self.BS(self.S, self.K, self.T, self.impvol - h, self.r, self.q)
        return (p1-p2)/(2*h*100)

    def theta(self):
        """
        Calculate the theta of the option.
        """
        h = THETA_DIFFERENTIAL
        p1 = self.BS(self.S, self.K, self.T + h, self.impvol, self.r, self.q)
        p2 = self.BS(self.S, self.K, self.T - h, self.impvol, self.r, self.q)
        return (p1-p2)/(2*h*365)

    def rho(self):
        """
        Calculate the rho of the option.
        """
        h = RHO_DIFFERENTIAL
        p1 = self.BS(self.S, self.K, self.T, self.impvol, self.r + h, self.q)
        p2 = self.BS(self.S, self.K, self.T, self.impvol, self.r - h, self.q)
        return (p1-p2)/(2*h*100)
