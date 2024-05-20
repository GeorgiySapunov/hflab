#!/usr/bin/env python3


# |------------------+-------------+---------------------|
# | BPG              | SHF 12105 A | 192.168.0.202:50000 |
# | DAC              | SHF 614 C   | 192.168.0.204:50000 |
# | CLCKSRC          | SHF 78122 B |                COM6 |
# | PAM4 Multiplexer | SHF 616 C   |  192.168.0.200:5000 |
# | EA               | SHF 11104 A |                     |
# | Mainframe        | SHF 10000 A |                     |
# |------------------+-------------+---------------------|

# BPG
BPG:SWINFO=?
BPG:ID=PREFIX: BPG,SN:51001,PN:12105,OPTION: 0,VER: 1.0,REV:A;
BPG: SWINFO=?;
SHF:ID=?;
SHFID=PREFIX:BPG,SN:51001,PN: 12105,OPTION: 0,VER:1.0,REV:A;
BPG: SWINFO-KERNEL:Linux 4.0.0 #56 SMP PREEMPT Thu Jul 27 15:41:14 CEST 2017,SERVER:1.0.27;
BPG: VCSINFOMASTER=?
BPG:VCSINFOMASTER=REPOSITORY:https://vc/hg/com_pn89132,BRANCH:default,CHANGESET:959399fcc745,DATE:2018-08-31 09:10:30,HOST:ap138.shf-berlin.local,MODIFIED:NO,PATH:/home/schunke/projects/xilinx/com_pn89132_2016_4,URL:N/A,TAG:1.7.1,USER:N/ A VERSION:N/A,XILINXISE:N/A,XILINXEDK:N/A,XILINXRELEASE:N/A,VIVADO:2016.4.0;
BPG: VCSINFOSLAVE=?;
BPG:VCSINFOSLAVE=REPOSITORY:https://vc/hg/com_pn89133_shf12105,BRANCH:default,CHANGESET:477dd4e552ed,DATE: 2018-11-23 10:13:07,HOST:ap138.shf-berlin.local,MODIFIED:NO,PATH:/home/schunke/projects/xilinx/com_pn89133_shf12105_kintex,URL:N/A,TAG: 1.6.0,USER:N/A,VERSION:N/A,XILINXISE:N/A,XILINXEDK:N/A,XILINXRELEASE:N/A,VIVADO:2016.4.0;
SHF:ID=?;
SHF:ID=PREFIX:BPG,SN:51001,PN:12105,OPTION:0,VER:1.0,REV:A;
BPG: SWINFO=?;
BPG: SWINFO=KERNEL:Linux 4.0.0 #56 SMP PREEMPT Thu Jul 27 15:41:14 CEST
2017,SERVER:1.0.27;
BPG:PATTERN= Channel1:None,Channel2:None,Channel3:None,Channel4:None,Channel5:None,Channel6:None;
BPG: PATTERIN=FAIL;
BPG: PREEMPHASIS=ENABLE: OFF;
BPG: PREEMPHASIS=ENABLE:OFF;
BPG: FIRFILTER=ENABLE:OFF;
BPG: FIRFILTER=ENABLE:OFF;
BPG: VCSINFOMASTER=?;
BPG: VCSINFOMASTER=REPOSITORY:https://vc/hg/com_pn89132,BRANCH:default,CHANGESET:959399fcc745,DATE:2018-08-31 09:10:30,HOST:ap138.shf-berlin.local,MODIFIED:NO,PATH:/home/schunke/projects/xilinx/com_pn89132_2016_4,URL:N/A,TAG:1.7.1,USER:N/ A,VERSION:N/A,XILINXISE:N/A,XILINXEDK:N/A,XILINXRELEASE:N/A,VIVADO:2016.4.0;
BPG: VCSINFOSLAVE=?;
BPG:VCSINFOSLAVE=REPOSITORY:https://vc/hg/com_pn89133_shf12105,BRANCH:default,CHANGESET:477ddd4e552ed,DATE:2018-11-23 10:13:07,HOST:ap138.shf-berlin.local,MODIFIED:NO,PATH:/home/schunke/projects/xilinx/com_pn89133_shf12105_kintex, URL:N/A,TAG: 1.6.0,USER:N/A,VERSION:N/A,XILINXISE:N/A,XILINXEDK:N/AXILINXRELEASE:N/A, VIVADO:2016.4.0;
BPG:ID=?
BPG::D=PREFIX:BPG,SN: 51001,PN:12105,OPTION:0,VER:1.0,REV:A;
BPG: ALLOWNEWCONNECTIONS=?
BPG: ALLOWNEWCONNECTIONS=ON;
BPG SENDWRGON
BPG: HELP=PATTERN;

# BPG: HELP="PATTERN=(ALL,Channel3x7,Channel1x5,Channel4,Channel5,Channel6,Channel7,Channel1,Cha
SHF:ID=?
SHFID=PREFIX:BPG,SN:51001,PN:12105,OPTION:0,VER:1.0,REV:A;
BPG: SWINFO=?
BPG:SWINFO=KERNEL:Linux 4.0.0 #56 SMP PREEMPT Thu Jul 27 15:41:14 CEST 2017,SERVER:1.0.27;
BPG: VCSINFOMASTER=?;
BPG:VCSINFOMASTER=REPOSITORY:https://vc/hg/com_pn89132,BRANCH:d
BPG:SWINFO=KERNEL:Linux 4.0.0 #56 SMPPREEMPT Thu Jul 27 15:41: 09:10:30,HOST:ap138.shf-berlin.local,MODIFIED:NO,PATH:/home/schunke/projects/xilinx/com_pn89132_2016_4,URL:N/A,TAG:1.7.1,USER:N/ A,VERSION:N/A,XILINXISE:N/A,XILINXEDK:N/A,XILINXRELEASE:N/A,VIVADO: 2016.4.0
BPG: VCSINFOSLAVE=?;
BPG:VCSINFOSLAVE=REPOSITORY:https://vc/hg/com_pn89133_shf12105,BRANCH:default,CHANGESET:477dd4e552ed,DATE:2018-11-23 10:13:07,HOST:ap138.shf-berlin.local,MODIFIED:NO,PATH:/home/schunke/projects/xilinx/com_pn89133_shf12105_kintexURL:N/A,TAG: 1.6.0,USER:N/A,VERSION:N/A,XILINXISE:N/A,XILINXEDK:N/A,XILINXRELEASE:N/A,VIVADO:2016.4.0;
BPG: SENDCLIENTS=ON
BPG: SENDCLIENTS=ON;
BPG: CLIENTS=?
BPG: CLIENTS= Client 3 connected to 192.168.0.10 port: 49722;
BPG: CAPABILITIES=?;
BPG:CAPABILITIES=ERRORINJECTION:[OFF|1e-10…0.01] OUTPUTS:ALL|Channel3x7|Channel1x5|Channel4|Channel5|Channel6|Channel7] Channel1|Channel2|Channel3|Channel8|Channel4x8|Channel2x6,PAM|4PATTERN:(1]+/-SHIFT,PATTERN:[3[PRBS7|PRBS9|PRBS10|PRBS11| PRBS13|PRBS15|PRBS20|PRBS20_ITU|PRBS23|PRBS31|USER|USERPATTERN_FIXEDLENGTH|PAMX|SQUARE1|SQUARE2|SQUARE4|SQUARE8| SQUARE16|SQUARE32|WF_PRBS7|WF_PRBS9|WF_PRBS10|WF_PRBS11|WF_PRBS13|WF_PRBS15|WF_PRBS20|WF_PRBS23|WF_PRBS31|WF_USER] +/-SHIFT,PHYSOUTPUTS:ALL|Channel4|Channel5|Channel6|Channel7|Channel1|Channel2|Channel3|Channel8,SELECTABLECLOCK:4/8|16|32|64| 128|256|512|1024;
BPG: CAPABILITIES=AUTOSKEWOUTPUTS:?;
BPG:CAPABILITIES=FAIL;
BPG: HELP=PATTERN;
BPG:HELP="PATTERN=(ALL,Channel3x7,Channel1x5,Channel4,Channel5,Channel6,Channel7,Channel1,Cha
BPG: PATTERN=?

BPG: PATTERN=Channel1:PRBS7,Channel2:PRBS7,Channel3:PRBS7,Channel4:PRBS7,Channel5:PRBS7,Cha
BPG: HELP=ERRORINJECTION;
BPG: HELP="ERRORINJECTION=(ALL,Channel3x7,Channel1x5,Channel4,Channel5,Channel6,Channel7,Chan
BPG: OUTPUTLEVEL=0;
BPG: OUTPUTLEVEL=0;
BPG: AMPLITUDERANGES=?
BPG:AMPLITUDERANGES=Channel1:40 mV…730 mV,Channel2:50 mV…730 mV,Channel3:50 mV…725 mV,Channel4:50 mV…740 mV,Channel5:50 mV…760 mV,Channel6:50 mV…745 mV,Channel7:50 mV…750 mV,Channel8:50 mV…750 mV;
BPG: SENDAMPLITUDERANGES=ON;
BPG: SENDAMPLITUDERANGES=ON;
BPG: AMPLITUDE=?;
BPG:AMPLITUDE= Channel1:650 mV,Channel2:650 mV,Channel3:650 mV,Channel4:650 mV,Channel5:650 mV,Channel6:650 mV,Channel7:500 mV,Channel8:500 mV;
BPG:SENDAMPLITUDE=ON;
BPG:SENDAMPLITUDE=ON;
BPG:OUTPUT=?;
BPG:OUTPUT=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF:
BPG: SENDOUTPUT=ON;
BPG: SENDOUTPUT=ON;
BPG: PATTERN=?;
BPG:PATTERN=Channel1:PRBS7,Channel2:PRBS7,Channel3:PRBS7,Channel4:PRBS7,Channel5:PRBS7,Cha
BPG:SENDPATTERN=ON;
BPG:SENDPATTERN=ON
BPG:SKEWRANGES=?
BPG:SKEWRANGES= Channel1:-33.1 ps.37.5 ps, Channel2:-37.1 ps.33.8 ps, Channel3:-30.2 ps.40 ps, Channel4:-35.5 ps…34.5 ps, Channel5:-26.4 ps, 32.4 ps Channel6:-31.2 ps,33 ps Channel7:-31.6 ps,32.4 ps,Channel8:-33.9 ps, 32.9 ps;
BPG: SENDSKEWRANGES=ON;
BPG: SENDSKEWRANGES=ON;
BPG. SKEW=?

BPG: SKEW=Channel1:7 ps,Channel2:6 ps,Channel3:7 ps,Channel4:5 ps,Channel5:1 ps,Channel6:4 ps,Channel7:0 s,Channel8:0 s;
BPG: SENDSKEW=ON;
BPG:SENDSKEW=ON;
BPG:ERRORINJECTION=?;
BPG:ERRORINJECTION=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF;
BPG: SENDERRORINJECTION=ON;
BPG: SENDERRORINJECTION=ON;
BPG: DUTYCYCLEADJUST=?
BPG:DUTYCYCLEADJUST=Channel1:0,Channel2:0,Channel3:0,Channel4:0,Channel5:0,Channel6:0,Channel7:0,Channel8:0;
BPG: SENDDUTYCYCLEADJUST=ON;
BPG: SENDDUTYCYCLEADJUST=ON;
BPG: BITRATEDIVIDER=?;
BPG: BITRATEDIVIDER=Channel1:1, Channel2:1, Channel3:1, Channel4:1, Cha, nel5:1, Channel6:1,Channel7:1, Channel8:1;
BPG:SENDBITRATEDIVIDER=ON;
BPG: SENDBITRATEDIVIDER=ON;
BPG: BITRATE=?
BPG: BITRATE= Channel1:12673348720 nolock,Channel2:12673348720 nolock,Channel3:12673348720 nolock,Channel4:12673348720 nolock,Channel5:12673348720 nolock,Channel6:12673348720 nolock,Channel7:12673348720 nolock,Channel8:12673348720 nolock,Channel1x5:25346697440 nolock,Channel2x6:25346697440 nolock,Channel3x7:25346697440 nolock,Channel4x8:25346697440 nolock;
BPG: SKEWRANGES=Channel1:?;
BPG: SKEWRANGES=Channel1:-33.1 ps…37.5 ps;
BPG: SKEWRANGES=Channel2:?;
BPG: SKEWRANGES=Channel2:-37.1 ps33.8 ps;
BPG: SKEWRANGES=Channel3:?;
BPG:SKEWRANGES=Channel3:-30.2 ps…40 ps;
BPG: SKEWRANGES=Channel4:?;
BPG: SKEWRANGES=Channel4:-35.5 ps…34.5 ps;
BPG: SKEWRANGES=Channel5:?;
BPG:SKEWRANGES=Channel5:-26.4 ps…32.4 ps;
BPG: SKEWRANGES=Channel6:?;
BPG: SKEWRANGES=Channel6;-31,2 ps.33 ps;
BPG:SKEWRANGES=Channel7
BPG:SKEWRANGES=Channel7:-31.6 ps.32.4 ps;
BPG: SKEWRANGES=Channel8:?
BPG:SKEWRANGES= Channel8:-33.9 ps…32.9 ps;
BPG:SENDBITRATE=ON;
BPG: SENDBITRATE=ON;

BPG: OUTPUTINFO=Channel1x5:?
BPG: OUTPUTINFO=Channel1x5:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static
constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: USERPATTERNLOADFILE=Channel1x5.STATUS:?;
BPG:USERPATTERNLOADFILE=Channel1x5.STATUS:Idle;
BPG: OUTPUTINFO=Channel1x5:?
BPG:OUTPUTINFO=Channel1x5:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static
constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: OUTPUTINFO=Channel2x6:?
BPG:OUTPUTINFO=Channel2x6:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes, total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: USERPATTERNLOADFILE= Channel2x6.STATUS:?;
BPG: USERPATTERNLOADFILE=Channel2x6.STATUS:Idle;
BPG: OUTPUTINFO=Channel2x6:?
BPG:OUTPUTINFO=Channel2x6:virtual 128 Gbit's fixed skew, multiplex parent output: NONE, userpattern capable: YES, user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: OUTPUTINFO=Channel3x7:?
BPG: OUTPUTINFO= Channel3x7:virtual 128 Gbit's fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG:USERPATTERNLOADFILE=Channel3x7.STATUS:?;
BPG: USERPATTERNLOADFILE=Channel3x7.STATUS:Idle
BPG: OUTPUTINFO=Channel3x7:?
BPG:OUTPUTINFO=Channel3x7:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capablYES, user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG:OUTPUTINFO=Channel4x8:?
BPG:OUTPUTINFO=Channel4x8:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: USERPATTERNLOADFILE=Channel4x8.STATUS:?;
BPG: USERPATTERNLOADFILE= Channel4x8.STATUS:Idle;
BPG: OUTPUTINFO= Channel4x8:?
BPG:OUTPUTINFO=Channel4x8:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES, user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG:OUTPUTINFO= Channel1x5;?

BPG:OUTPUTINFO=Channel1x5:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: USERPATTERNLOADFILE=Channel1x5.STATUS:?;
BPG: USERPATTERNLOADFILE=Channel1x5.STATUS:Idle;
BPGOUTPUTINFO=Channel1x5:?
BPG:OUTPUTINFO=Channel1x5:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES, user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: OUTPUTINFO=Channel2x6:?
BPG:OUTPUTINFO=Channel2x6:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern cappble: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: USERPATTERNLOADFILE=Channel2x6.STATUS:?;
BPG: USERPATTERNLOADFILE=Channel2x6.STATUS:Idle;
BPG: OUTPUTINFO=Channel2x6:?
BPG:OUTPUTINFO=Channel2x6:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG:OUTPUTINFO=Channel3x7:?
BPG:OUTPUTINFO=Channel3x7:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment 2 GBytes;
BPG:USERPATTERNLOADFILE= Channel3x7.STATUS:?;
BPG: USERPATTERNLOADFILE=Channel3x7.STATUS:Idle;
BPG: OUTPUTINFO=Channel3x7:?
BPG: OUTPUTINFO=Channel3x7:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: OUTPUTINFO=Channel4x8:?
BPG:OUTPUTINFO=Channel4x8:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES, user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes,total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: USERPATTERNLOADFILE=Channel4x8.STATUS:?;
BPG: USERPATTERNLOADFILE=Channel4x8.STATUS:Idle;
BPG: OUTPUTINFO=Channel4x8:?
BPG:OUTPUTINFO=Channel4x8:virtual 128 Gbit/s fixed skew, multiplex parent output: NONE, userpattern capable: YES,user pattern static constraints (min/step/max): 64 Bytes/64 Bytes/2 GBytes total free user pattern memory: 2 GBytes,largest free segment: 2 GBytes;
BPG: AVAILABLEBITRATEDIVIDERS=ALL:
BPG:AVAILABLEBITRATEDIVIDERS=Channel1:[1|2|4], Channel2:[|2[4], Channel3:[|2|4], Channel4:[12|4], Channel5:[12|4], Channel6:[1|2] 4] Channel7:[1|2|4] Channel8:[1|2|4];
BPG:CLOCKINPUT-?
BPG.CLOCKINPUT=FULL
BPG: SENDCLOCKINPUT=ON;
BPG: SENDCLOCKINPUT+ON;
BPG: SELECTABLEOUTPUT=?
BPG: SELECTABLEOUTPUT=SELECTABLECLOCK;

BPG: SELECTABLE OUTPUT=SELECTABLE CLOCK;
BPG: SEND SELECTABLE OUTPUT=ON;
BPG: SEND SELECTABLE OUTPUT=ON;
BPG: SELECTABLE CLOCK=?;
BPG: SELECTABLE CLOCK=4;
BPG: SEND SELECTABLE CLOCK=ON;
BPG: SEND SELECTABLE CLOCK=ON;
BPG: SEND FIR FILTER=ON;
BPG: SEND FIR FILTER=ON;
BPG: USERSETTINGS=?;
BPG: USER SETTINGS=SCC.GRAY CODING:12,SCC.PATTERN TYPE:PRBS7;
FPGA:FIR FILTER=?
FPGA:FIR FILTER=DAC:SCC,ENABLE:OFF,H0:0,H1:1,H2:0,H3:0,FUNCTION:1"y+0,G 0:PRBS7,G1:PRBS7;
BPG:USER SETTINGS=SCC.PATTERN TYPE:PRBS7
BPG:USER SETTINGS=SCC.PATTERN TYPE:PRBS7;
BPG: SEND DAC DRIVER=ON;
BPG: SEND DAC DRIVER=ON;
BPG:DAC DRIVER=?;
FPGA:DAC DRIVER=SCC.NUMBITS:6,SCC.ENCODING:UNIPOLAR,SCC.FOR_PAM ORDER: 2,SCC.INPUT 0:Channel 1,SCC.INPUT 1:Channel 2,SCC.INPUT 2:Channel 3,SCC.INPUT 3:Channel 4,SCC.INPUT 4

BPG:SEND SPAM LEVELS=ON;
BPG: SEND SPAM LEVELS=ON;
BPG: USERSETTINGS=?
BPG:USER SETTINGS=SCC.GRAY CODING:12,SCC.PATTERN TYPE:PRBS7;
BPG: PAM LEVELS=?;
BPG: PAM LEVELS=SCC.PAM ORDER:2,SCC.L 0:0%,SCC IN:100%;
BPG:SEND PRE EMPHASIS=On;
BPG:SEND PRE EMPHASIS=ON;
BPG: PREEMPHASIS=?;
BPG:PREEMPHASIS=DAC:SCC,ENABLE:OFF,PAM LEVELS:NONE,TAP:100%,TAP 1:M,TAPE 2:0%, TAP 3:0%;
BPG: SEND POWER CONNECTOR=ON;
BPG: SEND POWER CONNECTOR=ON;
BPG: POWER CONNECTOR=Right+12V.SWITCH:?;
BPG: POWER CONNECTOR=Right+12V.SWITCH:OFF;
BPG: POWER CONNECTOR=Right+5V.SWITCH:?;
BPG: POWER CONNECTOR=Right+5V.SWITCH: OFF;
BPG: POWER CONNECTOR=Right-5V.SWITCH:?;
BPG: POWER CONNECTOR=Right-5V.SWITCH:OFF;

BPG: POWERCONNECTOR=Left+12V.SWITCH:?;
BPG: POWERCONNECTOR=Left+12V.SWITCH: OFF;
BPG: POWERCONNECTOR=Left+5V.SWITCH:?;
BPG: POWERCONNECTOR=Left+5V.SWITCH:OFF;
BPG: POWERCONNECTOR=Left-5V.SWITCH:?;
BPG: POWERCONNECTOR=Left-5V.SWITCH: OFF;
BPG: BITRATE=Channel1:12674102856 nolock,Channel2:12674102856 nolock,Channel3:12674102856 nolock,Channel4:12674102856 nolock,Channel5:12674102856 nolock,Channel6:12674102856 nolock,Channel7:12674102856 nolock,Channel8:12674102856 nolock;
BPG: FIRFILTER=GO:IPRBS7,G1:IPRBS7;
BPG: FIRFILTER=G0:!PRBS7,G1:!PRBS7;
BPG: CAPABILITIES=?;
BPG:CAPABILITIES=ERRORINJECTION:[OFF|1e-10…0.01],OUTPUTS:ALL|Channel3x7|Channel1x5|Channel4|Channel5|Channel6|Channel7| Channel1|Channel2|Channel3|Channel8|Channel4x8|Channel2x6,PAM4PATTERN:[![]+/-SHIFT,PATTERN:[][PRBS7|PRBS9|PRBS10|PRBS11| PRBS13|PRBS15|PRBS20|PRBS20_ITU|PRBS23|PRBS31|USER|USERPATTERN_FIXEDLENGTH|PAMX|SQUARE1|SQUARE2|SQUARE4|SQUARE8| SQUARE16|SQUARE32|WF_PRBS7|WF_PRBS9|WF_PRBS10|WF_PRBS11|WF__PRBS13|WF_PRBS15|WF_PRBS20|WF_PRBS23|WF_PRBS31|WF_USER] +/-SHIFT,PHYSOUTPUTS:ALL|Channel4|Channel5|Channel6|Channel7|Channel1|Channel2|Channel3|Channel8,SELECTABLECLOCK:4|8|16|32| 64|128|256|512|1024;
BPG: SENDWORDFRAME=ON;
BPG: WORDFRAME=?
BPG: SENDWORDFRAME=ON;
BPG: WORDFRAME=PRBS31,PulseWidthRatio:0.5;
BPG:BITRATE= Channel1:12673516136 nolock,Channelf 12673516136 nolock Channel3:12673516136 nolock Channel4:12673516136 nolock,Channel5:12673516136 nolock,Channel6:12673316136 nolock,Channel7:12673516136 nolock,Channel8:12673516136 nolock;
BPG: BITRATE=Channel1:12672964740 nolock,Channel2:12672964740 nolock,Channel3:12672964740 nolock,Channel4:12672964740 nolock,Channel5:12672964740 nolock,Channel6:12672964740 nolock,Channel7:12672964740 nolock,Channel8:12672964740 nolock;
BPG:SKEWRANGES=Channel1:;
BPG: BITRATE= Channel1:12672849032 nolock,Channel2:12672849032 nolock,Channel3:12672849032 nolock,Channel4:12672849032 nolock,Channel5:12672849032 nolock,Channel6:12672849032 nolock,Channel7:12672849032 nolock,Channel8:12672849032 nolock;
BPG: SKEWRANGES= Channel1:-33.1 ps.37.5 ps;
BPG: SKEWRANGES= Channel2?
BPG:SKEWRANGES=Channel2:-37,1 ps…33.8 ps;
BPG:SKEWRANGES=Channel3:
BPG:SKEWRANGES=Channel3:-30.2 ps.40 ps;
BPG:SKEWRANGES=Channel4:
BPG:SKEWRANGES-Channel4: 35.5 35.5 ps, 34,5 ps;

BPG: SKEWRANGES=Channel5:?;
BPG: SKEWRANGES=Channel5:-26.4 ps…32.4 ps;
BPG: SKEWRANGES=Channel6:?;
BPG: SKEWRANGES= Channel6:-31.2 ps…33 ps;
BPG: SKEWRANGES=Channel7:?;
BPG: SKEWRANGES=Channel7:-31.6 ps…32.4 ps;
BPG: SKEWRANGES=Channel8:?;
BPG: SKEWRANGES=Channel8:-33.9 ps…32.9 ps;
BPG: BITRATE=Channel1:12672323748 nolock,Channel2:12672323748 nolock,Channel3:12672323748 nolock,Channel4:12672323748 nolock, Channel5:12672323748 nolock,Channel6:12672323748 nolock,Channel7:12672323748 nolock,Channel8:12672323748 nolock;
BPG: BITRATE=Channel1:12671871676 nolock,Channel2:12671871676 nolock,Channel3:12671871676 nolock,Channel4:12671871676 nolock,Channel5:12671871676 nolock,Channel6:12671871676 nolock,Channel7:12671871676 nolock,Channel8:12671871676 nolock;
BPG: BITRATE=Channel1:12671508688 nolock,Channel2:12671508688 nolock,Channel3:12671508688 nolock,Channel4:12671508688 nolock,Channel5:12671508688 nolock,Channel6:12671508688 nolock,Channel7:12671508688 nolock,Channel8:12671508688 nolock;
BPG: SKEWRANGES=Channel1:?;
BPG: SKEWRANGES=Channell1:-33.1 ps37.5 ps;
BPG:SKEWRANGES=Channel2:?
BPG: SKEWRANGES=Channel2:-37.1 ps.33.8 ps;
BPG: SKEWRANGES=Channel3:?;
BPG:SKEWRANGES= Channel3:-30.2 ps.40 ps;
BPG: SKEWRANGES=Channel4:?
BPG: SKEWRANGES=Channel4:-35.5 ps…34.5 ps;
BPG:SKEWRANGES=Channel5
BPG: SKEWRANGES=Channel5:-26.4 ps…32.4 ps;
BPG: SKEWRANGES=Channel6:?;
BPG: SKEWRANGES=Channel6:-31.2 ps…33 ps;
BPG:SKEWRANGES=Channel7:?
BPG:SKEWRANGES=Channel7:-31.6 ps…32.4 ps
BPG:SKEWRANGES=Channel8:?;
BPG:SKEWRANGES=Channel8:-33.9 ps.32,9 ps;
BPG:BITRATE= Channel1:12671744708 nolockChannel2:12671744708 nolock Channel3:12671744708 nolock Channel4:12671744708 nolock,Channel5:12671744708 nolock,Channel6:12671744708 nolock,Channel7:12671744708 nolock,Channel8:12671744708 nolock
BPG: BITRATE Channel1:12670969068 nolockChannel2:12670969068 nolock,Channel3:12670969068 nolock Channel4:12670969068 nolock Channel5:12670969068 nolock Channel6:12670969068 nolock Channel7:12670969068 nolock Channel8:12670969068 nolock;
BPG:BITRATE- Channel1:12670741240 nolock,Channel2:12670741240 nolock,Channel3:12670741240 nolock,Channel4:12670741240 nolock Channel5:12670741240 nolock Channel6:12670741240 nolock Channel7:12670741240 nolock Channel8:12670741240 nolock

BPG: SKEWRANGES=Channel1:?;
BPG:SKEWRANGES=Channel1:-33.1 ps…37.5 ps;
BPG: SKEWRANGES=Channel2:?
BPG: SKEWRANGES=Channel2:-37.1 ps…33.8 ps;
BPG: SKEWRANGES=Channel3:?;
BPG:SKEWRANGES=Channel3:-30.2 ps…40 ps;
BPG: SKEWRANGES=Channel4:?;
BPG:SKEWRANGES=Channel4:-35.5 ps…34.5 ps;
BPG: SKEWRANGES=Channel5:?;
BPG:SKEWRANGES=Channel5:-26.4 ps…32.4 ps;
BPG: SKEWRANGES= Channel6:?;
BPG: SKEWRANGES=Channel6:-31.2 ps…33 ps;
BPG: SKEWRANGES=Channel7:?;
BPG: SKEWRANGES=Channel7:-31.6 ps…32.4 ps;
BPG: SKEWRANGES=Channel8:2;
BPG:SKEWRANGES=Channel8:-33.9 ps…32.9 ps;
BPG: BITRATE= Channel1:12670341900 nolock,Channel2:12670341900 nolock Channel3:12670341900 nolock Channel4:12670341900 nolock,Channel5:12670341900 nolock,Channel6:12670341900 nolock,Channel7:12670341900 nolock, Channel8:12670341900 nolock;
BPG: BITRATE=Channel1:12669650224 nolock,Channel2:12669650224 nolock, Channel3:12669650224 nolock,Channel4:12669650224 nolock,Channel5:12669650224 nolock,Channel6:12669650224 nolock,Channel7:12669650224 nolock,Channel8:12669650224 nolock;
BPG: SKEWRANGES=Channel1:?;
BPG:SKEWRANGES=Channel1:-33.1 ps…37.5 ps;
BPG: SKEWRANGES=Channel1:?:
BPG: SKEWRANGES=Channel?:?
BPG: SKEWRANGES=Channel2:-37.1 ps33.8 ps;
BPG:SKEWRANGES=Channel3:?;
BPG:SKEWRANGES=Channel3:-30.2 ps, 40 ps
BPG:SKEWRANGES= Channel4:
BPG:SKEWRANGES- Channel4:-35,5 ps…34,5 ps:
BPG: SKEWRANGES=Channel5:?
BPG: SKEWRANGES=Channel5:-26.4 ps…32.4 ps;
BPG:SKEWRANGES= Channel6:?
BPG:SKEWRANGES= Channel6:-31.2 ps…33 ps;

BPG: SKEWRANGES=Channel7:?;
BPG: SKEWRANGES=Channel7:-31.6 ps…32.4 ps;
BPG: SKEWRANGES=Channel8:?;
BPG: SKEWRANGES=Channel8:-33.9 ps…32.9 ps;
BPG: BITRATE=Channel1:9749946076 nolock,Channel2:9749946076 nolock,Channel3:9749946076 nolock,Channel4:9749946076 nolock,Channel5:9749946076 nolock,Channel6:9749946076 nolock,Channel7:9749946076 nolock,Channel8:9749946076 nolock;
BPG: SKEWRANGES=Channel1:?
BPG:SKEWRANGES=Channel1:-33.1 ps…37.5 ps;
BPG: SKEWRANGES=Channel2:?;
BPG: SKEWRANGES=Channel2:-37.1 ps…33.8 ps;
BPG: SKEWRANGES=Channel3:?;
BPG: SKEWRANGES=Channel3:-30.2 ps…40 ps;
BPG: SKEWRANGES=Channel4:?;
BPG: SKEWRANGES=Channel4:-35.5 ps…34.5 ps;
BPG:SKEWRANGES=Channel5:?;
BPG: SKEWRANGES=Channel5:-26.4 ps…32.4 ps;
BPG: SKEWRANGES=Channel6:?;
BPG:SKEWRANGES=Channel6:-31.2 ps…33 ps;
BPG: SKEWRANGES=Channel7:?;
BPG: SKEWRANGES=Channel7:-31.6 ps…32.4 ps;
BPG: SKEWRANGES=Channel8:?;
BPG: SKEWRANGES=Channel8:-33.9 ps…32.9 ps;
BPG: BITRATE=Channel1:9714288656 nolock,Channel2:9714288656 nolock,Channel3:9714288656 nolock,Channel4:9714288656 nolock,Channel5:9714288656 nolock,Channel6:9714288656 nolock,Channel7:9714288656 nolock,Channel8:9714288656 nolock;
BPG: BITRATE=Channel1:9711030456 nolock, Channel2:9711030456 nolock Channel3:9711030456 nolock,Channel4:9711030456 nolock Channel5:9711030456 nolock Channel6:9711030456 nolock Channel7:9711030456 nolockChannel8:9711030456 nolock
BPG: BITRATE=Channel1:9707833180 nolock,Channel2:9707833180 nolock,Channel3:9707833180 nolock,Channel4:9707833180 nolock,Channel5:9707833180 nolock,Channel6:9707833180 nolock, Channel7:9707833180 nolockChannel8:9707833180 nolock;
BPG: SKEWRANGES=Channel1:?;
BPG:SKEWRANGES=Channel1:-33.1 ps…37.5 ps;
BPG: SKEWRANGES=Channel2:?;
BPG:SKEWRANGES=Channel2:-37.1 ps.33.8 ps;
BPG:SKEWRANGES=Channel3:?;
BPG: SKEWRANGES=Channel3:-30.2 ps…40 ps;
BPG: SKEWRANGES=Channel4:?;
