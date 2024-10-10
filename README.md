# Night Ride

## About

A Night Ride is a realtime computer animation (a [demo](https://en.wikipedia.org/wiki/Demoscene)).
A capture of the demo is available on YouTube: https://www.youtube.com/watch?v=OomA9W_N3Ik

For more information about Night Ride, see:
- https://www.ctrl-alt-test.fr/productions/night-ride/
- https://www.pouet.net/prod.php?which=98212

## Build & Tools

To build the demo, we use [Microsoft Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/).

Tips:

* For interactive development, build using the `Editor` configuration:
** Shaders are recompiled automatically when `Ctrl-S` is pressed.
** The Editor mode expects a file `moto.wav` that can be generated using the `wav_export` project.
Alternatively, comment out the `SOUND_ON` macro during development (PRs to improve this would be welcome).

* For performance, you may want to reduce the resolution in the file `shared.h` as the demo is very GPU-intensive.

* The demo uses a perprocessor to merge all shader input files. The output is stored in `preprocessed.scene.frag`.

* The minified shader is visible in `shaders.inl`. To analyze the minified code, you may want to check `shaders.min.frag` instead (it's minified, but indented and without renaming).

## Original README

                          Ctrl-Alt-Test & Alcatraz
                                   present
          
                                 Night Ride
          
                               An 8kB PC intro
                          Released at Deadline 2024

    Made by:
        - Jochen Feldkötter, aka Virgill
        - Julien Guertault, aka Zavie
        - Laurent Le Brun, aka LLB
        - Jan Scheurer, aka LJ

    Using:
        - Crinkler         https://github.com/runestubbe/Crinkler
        - Leviathan        https://github.com/armak/Leviathan-2.0
        - Shader Minifier  https://github.com/laurentlb/shader-minifier
        - Sointu           https://github.com/vsariola/sointu

    This audio-visual presentation is meant to be run  on a PC powered
    by Microsoft Windows Vista or newer, with an NVIDIA GPU equivalent
    to RTX3080 or higher.

    In the balance between binary size and performance, the latter was
    essentially sacrificed, hence the hardware requirements.  We might
    release a later version with a different tradeoff so the intro can
    be enjoyed smoothly on mid-range hardware.

    Inspiration was directly drawn from:
        - Drive   (2011, director of photography: Newton Thomas Sigel)
        - Heat          (1995, director of photography: Dante Spinott)
        - Tron Legacy (2010, director of photography: Claudio Miranda)
        - Nightcrawler  (2014, director of photography: Robert Elswit)
        - Eternal Apex               (https://acatalept.com, Acatalep)

    Special thanks to:
        - rubix, for the series of improvements to Shader Minifier.
        - Iñigo Quilez, for the invaluable trove of ressources.
        - Fabrice Neyret, for the code reviews over the years.
        - Gopher, for always being cheerful.

    We send our  greetings to  Approximate,  Cocoon, Gaspode, loopit,
    Mercury, 0b5vr, Poobrain, Razor 1911, Setsuko, Still,
    The Black Lotus, Titan, TokyoDemoFest, Turbo Knight and TRBL.

EOF
