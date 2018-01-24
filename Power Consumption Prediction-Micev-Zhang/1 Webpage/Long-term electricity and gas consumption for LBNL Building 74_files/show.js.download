jQuery(document).ready(function() {

    var extractTitleIdParam = function() {
        var matches = window.location.search.match(/title_id=([^&]+)/);
        return matches[1];
    };
    var title_id = extractTitleIdParam();
    var http = jQuery.ajax('data/datasets.json');

    http.success(function(data) {
        var datasets = data.datasets;
        var dset = datasets[title_id];

        jQuery('title').text(dset.title);
        jQuery('.post-title').text(dset.title);
        jQuery('.dataset-download a').attr('href', dset.link);
        jQuery('#summary .field-value').text(dset.summary);
        jQuery('#description .field-value').text(dset.description);
        jQuery('#value .field-value').text(dset.value);
    });

});
