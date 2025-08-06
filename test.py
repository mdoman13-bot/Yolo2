# test.py
import ffmpeg

# Example: read from stdin (pipe:0), output raw null stream
(
    ffmpeg
    .input('pipe:0')                     # input comes from STDIN
    .output('pipe:1', format='null')     # write to STDOUT, format 'null'
)

ffmpeg.run()  # this runs the constructed command line
