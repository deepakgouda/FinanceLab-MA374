#include <bits/stdc++.h>
// #include <koolplot.h>
using namespace std;

#define tol_i 1000
#define tol_d 1000.0
#define pb push_back
#define vecd vector <double>

double model(double M, double S0, double r, double T, double sig)
{
	double u, d, p, dt = (T/M), stockprice;
	u = exp(sig*sqrt(dt) + (r-sig*sig)*(dt));
	d = exp(-sig*sqrt(dt) + (r-sig*sig)*(dt));
	p = (exp(r*dt)- d)/(u-d);  
	map < pair<double,double> , unordered_set<double> > hash;
	
	hash[{0,0}].insert(S0);
	
	for(int i = 1; i<= (M); i++)
	{
		
		for(int j = 0; j <= i; j++)
		{
		
			stockprice = S0*(pow(u, i-j)*pow(d, j));
			int x = (int)(stockprice*tol_d);
			
			double y = x/tol_d;
			stockprice = y;
			int flag = 1;
			
			if(j > 0)
			{
				for(auto k: hash[{i-1, j-1 }])
				{
					int x1 = (int)(k*tol_d);
					double y1 = x1/tol_d;
					k = y1;
					if(stockprice <= k)
					{
						hash[{i, j}].insert(k);
					}
					else if(flag)
					{
						flag = 0;
						hash[{i, j}].insert(stockprice);
					}
					
				}
				// cout <<"\n";
			}
			if(j < i)
			{
				for( auto k: hash[{i-1, j }])
				{
					//cout <<hash[{i-1, j-1 }][k]<<"	";
					
					int x1 = (int)(k*tol_d);
					//cout << x<<" f\n";
					double y1 = x1/tol_d;
					k = y1;
					if(stockprice <= k)
					{
						hash[{i, j}].insert(k);
					}
					else if(flag)
					{
						flag = 0;
						hash[{i, j}].insert(stockprice);
					}
					
				}
				
			}
		}
	}

	map< pair<double , double>, double>point;
	for(int i = 0; i<= M; i++)
	{
		for(auto j: hash[{M,i}])
		{
			int val1 = S0*pow(u, M-i)*pow(d, i)*tol_i;
			int val2 = j*tol_i;
			double a,b;
			a = val1/tol_d;b = val2/tol_d;
			point[{a, b}] = -a + b;
			
		}
	}

	for(int i = M-1; i>= 0; i--)
	{
		for(int j = 0; j<=i; j++ )
		{
			for(auto k : hash[{i,j}])
			{
				int val1 = S0*pow(u, i-j)*pow(d, j)*tol_i;
				int val2 = k*tol_i;
				int val3 = S0*pow(u, i-j)*pow(d, j)*u*tol_i;
				int val4 = S0*pow(u, i-j)*pow(d, j)*d*tol_i;
				double a,b,c,d1;
				a = val1/tol_d;b = val2/tol_d;c = val3/tol_d;d1 = val4/tol_d;
				point[{a, b}] = (p*point[{c, max(c, b)}] + (1-p)*point[{d1, b}])*exp(-r*dt);
			}
		}
	}
	return point[{S0, S0}];
}

void display(vecd a)
{
	for(auto i : a)
		cout<<i<<" ";
	cout<<endl;
}

int main()
{
	vecd X, Y;
	double stock = 100.0, r = 0.08, T = 1, sig = 0.2, price;
	clock_t begin = clock();
	for(int i = 5; i <=50; i+=5)
	{
		price = model(i,stock,r,T,sig);
		X.pb((double)i);
		Y.pb(price);
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout<<"Time elapsed = "<<elapsed_secs<<endl;

	cout<<"X = ";
	display(X);
	cout<<"Y = ";
	display(Y);
	// Plotdata x = X, y = Y;
	// plot(x, y);
	return 0;
}