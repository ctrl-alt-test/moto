@pushd %~dp0
cd src\shaders
..\..\ShaderPreprocessor.exe fxaa.frag
..\..\ShaderPreprocessor.exe scene.frag
..\..\ShaderPreprocessor.exe postprocess.frag
..\..\ShaderPreprocessor.exe scene.vert
@popd

@if "%~1"=="--minify" (
    shader_minifier.exe -v -o src\shaders\shaders.inl src\shaders\preprocessed.scene.frag src\shaders\preprocessed.postprocess.frag
    shader_minifier.exe -o src\shaders\shaders.min.frag --no-renaming --format indented src\shaders\preprocessed.scene.frag src\shaders\preprocessed.fxaa.frag src\shaders\preprocessed.postprocess.frag
)
