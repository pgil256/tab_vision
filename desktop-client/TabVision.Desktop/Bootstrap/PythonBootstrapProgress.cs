namespace TabVision.Desktop.Bootstrap;

public sealed record PythonBootstrapProgress(
    string Stage,
    double Percentage,
    string Message
);
