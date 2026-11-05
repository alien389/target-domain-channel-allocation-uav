function run_visdrone_internal_eval_stage02(toolkitPath, gtPath, imgPath, resPath, outputDir, expectedCount, toolkitCommit)
% Stage02-only VisDrone evaluation wrapper for the locked inner-val subset.
% The upstream toolkit and all source annotations remain unmodified.

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

diaryPath = fullfile(outputDir, 'internal_matlab_output_stage02.txt');
diary(diaryPath);
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

addpath(toolkitPath);
addpath(fullfile(toolkitPath, 'utils'));

assert(exist(fullfile(toolkitPath, 'evalDET.m'), 'file') == 2, 'VisDrone evalDET.m is missing.');
assert(exist(gtPath, 'dir') == 7, 'Stage02 annotation subset directory is missing.');
assert(exist(imgPath, 'dir') == 7, 'Original VisDrone train image directory is missing.');
assert(exist(resPath, 'dir') == 7, 'Stage02 prediction directory is missing.');

nameImgs = findImageList(gtPath);
numImgs = length(nameImgs);
assert(numImgs == expectedCount, 'Stage02 internal validation image count mismatch.');
assert(expectedCount == 1294, 'This wrapper is restricted to Stage02 inner-val1294.');

[allgt, alldet] = saveAnnoRes(gtPath, resPath, imgPath, numImgs, nameImgs);
[AP, AP50, AP75, AR1, AR10, AR100, AR500] = calcAccuracy(numImgs, allgt, alldet);

metricValues = [AP, AP50, AP75, AR1, AR10, AR100, AR500];
assert(all(isfinite(metricValues)), 'One or more VisDrone metrics are non-finite.');

metrics = struct( ...
    'AP', double(AP), ...
    'AP50', double(AP50), ...
    'AP75', double(AP75), ...
    'AR1', double(AR1), ...
    'AR10', double(AR10), ...
    'AR100', double(AR100), ...
    'AR500', double(AR500));

metadata = struct( ...
    'evaluator', 'Official VisDrone DET MATLAB toolkit applied to Stage02 inner-val1294', ...
    'toolkit_path', toolkitPath, ...
    'toolkit_commit', toolkitCommit, ...
    'matlab_version', version, ...
    'evaluated_images', double(numImgs), ...
    'max_detections_per_image', 500, ...
    'metric_scale', 'percentage_0_to_100', ...
    'original_train_annotations_copied_read_only', true, ...
    'official_val548_used', false, ...
    'testdev_used', false);

result = struct('visdrone_metrics', metrics, 'metadata', metadata);
save(fullfile(outputDir, 'internal_visdrone_metrics_stage02.mat'), 'result', '-v7');

jsonText = jsonencode(result, 'PrettyPrint', true);
jsonPath = fullfile(outputDir, 'internal_visdrone_metrics_stage02.json');
fileId = fopen(jsonPath, 'w');
assert(fileId ~= -1, 'Could not open Stage02 metrics JSON.');
cleanupFile = onCleanup(@() fclose(fileId)); %#ok<NASGU>
fprintf(fileId, '%s', jsonText);

fprintf('STAGE02_VISDRONE_AP=%.12f\n', AP);
fprintf('STAGE02_VISDRONE_AP50=%.12f\n', AP50);
fprintf('STAGE02_VISDRONE_AP75=%.12f\n', AP75);
fprintf('STAGE02_VISDRONE_AR500=%.12f\n', AR500);
fprintf('STAGE02_VISDRONE_JSON=%s\n', jsonPath);
end
