import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=['zero','flat','raptor','full'];v=[39.3,66.9,66.9,75.2]
plt.bar(a,v,color=['#999','#4c78a8','#f58518','#54a24b']);plt.ylabel('QuALITY accuracy (%)');plt.title('Qwen3-4B, 145 questions, 8 articles')
for i,x in enumerate(v):plt.text(i,x+1,f'{x}',ha='center')
plt.savefig('chart.png',dpi=120)
