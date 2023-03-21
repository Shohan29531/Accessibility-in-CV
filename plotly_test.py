import plotly.express as px

# it will show the value of Z in a grid
z = [[.1, .3, .5, .7, .9],
 	[1, .8, .6, .4, .2],
 	[.2, 0, .5, .7, .9],
 	[.9, .8, .4, .2, 0],
 	[.3, .4, .5, .7, 1]]

fig = px.imshow(z, text_auto=True, aspect="auto")
fig.show()
## So, if we pass the correlation function, it will display the heatmap

## display(df.corr())
fig = px.imshow(df.corr(), text_auto=True, aspect="auto")

## to disable, text, set 'text_auto=False"
# fig = px.imshow(df.corr(), text_auto=False, aspect="auto")

fig.show()
