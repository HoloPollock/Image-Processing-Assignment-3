# Image compression
#
# You'll need Python 2.7 and must install these packages:
#
#   scipy, numpy
#
# You can run this *only* on PNM images, which the netpbm library is used for.
#
# You can also display a PNM image using the netpbm library as, for example:
#
#   python netpbm.py images/cortex.pnm


import sys, os, math, time, netpbm
import numpy as np


# Text at the beginning of the compressed file, to identify it


headerText = 'my compressed image - v1.0'



# Compress an image


def compress( inputFile, outputFile ):

  # Read the input file into a numpy array of 8-bit values
  #
  # The img.shape is a 3-type with rows,columns,channels, where
  # channels is the number of component in each pixel.  The img.dtype
  # is 'uint8', meaning that each component is an 8-bit unsigned
  # integer.
  img = netpbm.imread( inputFile ).astype('uint8')
  
  # Compress the image
  #
  # REPLACE THIS WITH YOUR OWN CODE TO FILL THE 'outputBytes' ARRAY.
  #
  # Note that single-channel images will have a 'shape' with only two
  # components: the y dimensions and the x dimension.  So you will
  # have to detect this and set the number of channels accordingly.
  # Furthermore, single-channel images must be indexed as img[y,x]
  # instead of img[y,x,1].  You'll need two pieces of similar code:
  # one piece for the single-channel case and one piece for the
  # multi-channel case.

  startTime = time.time()
  outputBytes = bytearray()
  diff_image = []

  # create diffrence array
  if len(img.shape) != 3: # single-channel image
    num_of_channels = 1
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        if x == 0: # first pixel
          diff_image.append(img[y][x])
        else:
          diff_image.append(img[y][x] - img[y][x-1])
  else: # not single-channel image
    num_of_channels = img.shape[2] 
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        for c in range(img.shape[2]): # include colour
          if x == 0: # first pixel
            diff_image.append(img[y][x][c])
          else:
            diff_image.append(img[y][x][c] - img[y][x-1][c])


  init_dict = {} # initialize dictionary
  dict_size = (255 * 2) 
  for i in range(dict_size + 1):
    init_dict[str(i - 255) + ","] = i # initialize all possible diffrences (no negative)
  dict_copy = init_dict.copy()
  sub_string = ""

  for element in diff_image:
    if dict_size >= 65536: # if larger than max allowed size
      dict_copy = init_dict.copy()
      dict_size = (255 * 2) 
    sub_string += str(element) + ","
    if not sub_string in dict_copy:
      dict_size += 1
      dict_copy[sub_string] = dict_size
      append_key = ",".join(sub_string.split(",")[:-2]) + ","
      byte_one = dict_copy[append_key] // (2**8) # get first byte value (from int)
      byte_two = dict_copy[append_key] & 0b11111111 # get second byte value (from int)
      outputBytes.append(byte_one)
      outputBytes.append(byte_two)
      sub_string = str(element) + ","
  byte_one = dict_copy[sub_string] // (2**8)
  byte_two = dict_copy[sub_string] & 0b11111111
  outputBytes.append(byte_one)
  outputBytes.append(byte_two)
    

       
  endTime = time.time()

  # Output the bytes
  #
  # Include the 'headerText' to identify the type of file.  Include
  # the rows, columns, channels so that the image shape can be
  # reconstructed.
  print num_of_channels
  outputFile.write( '%s\n'       % headerText )
  outputFile.write( '%d %d %d\n' % (img.shape[0], img.shape[1], num_of_channels) )
  outputFile.write( outputBytes )


  # Print information about the compression
  
  if len(img.shape) == 3:
    inSize  = img.shape[0] * img.shape[1] * img.shape[2]
    outSize = len(outputBytes)

    sys.stderr.write( 'Input size:         %d bytes\n' % inSize )
    sys.stderr.write( 'Output size:        %d bytes\n' % outSize )
    sys.stderr.write( 'Compression factor: %.2f\n' % (inSize/float(outSize)) )
    sys.stderr.write( 'Compression time:   %.2f seconds\n' % (endTime - startTime) )
  else:
    inSize = img.shape[0] * img.shape[1] * num_of_channels
    outSize = len(outputBytes)

    sys.stderr.write('Input size:         %d bytes\n' % inSize)
    sys.stderr.write('Output size:        %d bytes\n' % outSize)
    sys.stderr.write('Compression factor: %.2f\n' % (inSize / float(outSize)))
    sys.stderr.write('Compression time:   %.2f seconds\n' % (endTime - startTime))
  
  


# helper function to convert byte array into int values
# returns -1 if byte_iter has no next value

# Uncompress an image
def uncompress( inputFile, outputFile ):

  # Check that it's a known file
  
  if inputFile.readline() != headerText + '\n':
    sys.stderr.write( "Input is not in the '%s' format.\n" % headerText )
    sys.exit(1)
    
  # Read the rows, columns, and channels.  

  rows, columns, channels = [ int(x) for x in inputFile.readline().split() ]

  # Read the raw bytes.

  inputBytes = bytearray(inputFile.read())

  # Build the image
  #
  # REPLACE THIS WITH YOUR OWN CODE TO CONVERT THE 'inputBytes' ARRAY INTO AN IMAGE IN 'img'.

  startTime = time.time()
  def toInt():
    try:
      first = next(byteIter)
      second = next(byteIter) 
      output = (first << 8) | second
    except StopIteration:
      output = -1
    return output

  img = np.empty( [rows,columns,channels], dtype=np.uint8 )

  

  byteIter = iter(inputBytes)

  init_dict = {} 
  dict_size = (255 * 2)
  for i in range(dict_size + 1):
    init_dict[i] = str(i - 255) + "," # initialize all possible diffrences
  dict_copy = init_dict.copy()

  diff_image = []

  prev_key = toInt()
  prev_val = dict_copy[prev_key].split(",")[:-1]
  diff_image += prev_val
  current_key = 0

  while current_key != -1:
    if dict_size >= 65536: 
      dict_copy = init_dict.copy()
      dict_size = (255 * 2)
    current_key = toInt()
    if current_key == -1:
      break # no next value found, exit loop
    if not current_key in dict_copy:
      prev_val += [prev_val[0]]
      diff_image += prev_val
      dict_size += 1
      dict_copy[dict_size] = ",".join(prev_val) + ","
    else:
      current_val = dict_copy[current_key].split(",")[:-1]
      diff_image += current_val
      dict_size += 1
      dict_copy[dict_size] = ",".join(prev_val + [current_val[0]]) + ","
      prev_val = current_val



  index = 0
  
  if channels != 3:
    img = np.empty([rows, columns], dtype=np.uint8)
    for y in range(rows):
      img[y][0] = int(diff_image[index])
      index += 1
      for x in range(1, columns):
        img[y][x] = int(diff_image[index]) + img[y][x-1]
        index += 1
  else:
    for y in range(rows):
      for c in range(channels):
        img[y][0][c] = int(diff_image[index])
        index += 1
      for x in range(1, columns):
        for c in range(channels):
          img[y][x][c] = int(diff_image[index]) + img[y][x-1][c]
          index += 1

  endTime = time.time()

  # Output the image

  netpbm.imsave( outputFile, img )

  sys.stderr.write( 'Uncompression time: %.2f seconds\n' % (endTime - startTime) )

  

  
# The command line is 
#
#   main.py {flag} {input image filename} {output image filename}
#
# where {flag} is one of 'c' or 'u' for compress or uncompress and
# either filename can be '-' for standard input or standard output.


if len(sys.argv) < 4:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)

# Get input file
 
if sys.argv[2] == '-':
  inputFile = sys.stdin
else:
  try:
    inputFile = open( sys.argv[2], 'rb' )
  except:
    sys.stderr.write( "Could not open input file '%s'.\n" % sys.argv[2] )
    sys.exit(1)

# Get output file

if sys.argv[3] == '-':
  outputFile = sys.stdout
else:
  try:
    outputFile = open( sys.argv[3], 'wb' )
  except:
    sys.stderr.write( "Could not open output file '%s'.\n" % sys.argv[3] )
    sys.exit(1)

# Run the algorithm

if sys.argv[1] == 'c':
  compress( inputFile, outputFile )
elif sys.argv[1] == 'u':
  uncompress( inputFile, outputFile )
else:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)
