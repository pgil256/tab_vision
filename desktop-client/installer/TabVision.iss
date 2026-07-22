#ifndef SourceRoot
  #error SourceRoot must point to the prepared bundle directory.
#endif

#ifndef OutputDir
  #error OutputDir must point to the installer output directory.
#endif

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{84A498A6-43D6-4CC4-95DE-90E730D37053}
AppName=TabVision
AppVersion={#AppVersion}
AppPublisher=TabVision
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={localappdata}\Programs\TabVision
DefaultGroupName=TabVision
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=TabVision-{#AppVersion}-win-x64-setup
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupLogging=yes
UninstallDisplayIcon={app}\TabVision.Desktop.exe
WizardStyle=modern

[Files]
Source: "{#SourceRoot}\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceRoot}\bootstrap\*"; DestDir: "{app}\bootstrap"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceRoot}\bundle-manifest.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\TabVision"; Filename: "{app}\TabVision.Desktop.exe"
Name: "{autodesktop}\TabVision"; Filename: "{app}\TabVision.Desktop.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\TabVision.Desktop.exe"; Description: "Launch TabVision"; Flags: nowait postinstall skipifsilent
