; auto-generated by Sointu, editing not recommended
%ifndef PLAYER_INC
%define PLAYER_INC

%define SU_CHANNEL_COUNT        2
%define SU_LENGTH_IN_SAMPLES    6115200
%define SU_BUFFER_LENGTH        (SU_LENGTH_IN_SAMPLES*SU_CHANNEL_COUNT)

%define SU_SAMPLE_RATE          44100
%define SU_BPM                  90
%define SU_ROWS_PER_BEAT        4
%define SU_ROWS_PER_PATTERN     16
%define SU_LENGTH_IN_PATTERNS   52
%define SU_LENGTH_IN_ROWS       (SU_LENGTH_IN_PATTERNS*SU_PATTERN_SIZE)
%define SU_SAMPLES_PER_ROW      (SU_SAMPLE_RATE*60/(SU_BPM*SU_ROWS_PER_BEAT))
%define SU_SAMPLE_SIZE 4
%define SU_SAMPLE_RANGE 1.0
%define SU_SAMPLE_FLOAT

_su_symbols:
%ifdef MANGLED
	extern _su_render_song@4
%else ; MANGLED
    extern su_render_song
%endif ; MANGLED
	extern _su_load_gmdls
%define SU_LOAD_GMDLS

%endif ; PLAYER_INC
