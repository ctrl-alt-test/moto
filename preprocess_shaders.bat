@pushd %~dp0
cd src\shaders
..\..\ShaderPreprocessor.exe scene.frag
@popd

@if "%~1"=="--minify" (
    shader_minifier.exe -v -o src\shaders\shaders.inl src\shaders\preprocessed.scene.frag src\shaders\scene.vert src\shaders\fxaa.frag src\shaders\postprocess.frag
    shader_minifier.exe -o src\shaders\shaders.min.frag --no-renaming --format indented src\shaders\preprocessed.scene.frag src\shaders\scene.vert src\shaders\fxaa.frag src\shaders\postprocess.frag
)
