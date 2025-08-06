data_yield = np.genfromtxt('CropLoss_Corn_Flood_May_June_Updated_met_checked.csv', delimiter=',')
length = data_yield.shape[0]

index_all = np.arange(length)
output_area = np.zeros([index_all.shape[0]])

for i in index_all:
    output_area[i] = data_yield[i, 3]

MaxValue = np.max(output_area)
print("MaxValue =", MaxValue ) 
scaledMax = 5000
scaledMin = 0
k = (scaledMax - scaledMin) / (np.log10 (2 * MaxValue) - np.log10 ( MaxValue))
print("k =", k ) 
c =  -k *np.log10 (MaxValue)
print("c =", c )

np.save('c.npy',c)
np.save('k.npy',k)
np.save('MaxValue.npy',MaxValue)

output_area =k* np.log10(output_area + MaxValue) +c
print (np.max(output_area))
#Inverse f(n) = 10^((f(n) -c) / k) - MaxValue
output_area =np.power(10, (output_area - c)/k)-MaxValue
print("c =", np.max(output_area) )
print("c =", np.min(output_area) )



