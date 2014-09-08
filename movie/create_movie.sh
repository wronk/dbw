cd images/
# image resize is permanant
#mogrify -resize 875x396 *.png
convert -delay 8 -loop 0 *.png ../movie6_8ms.gif