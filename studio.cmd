@echo off
REM TabVision Studio launcher — record a take, get tabs.
REM Usage: studio.cmd            (just run it)
REM        studio.cmd -NoBrowser
powershell -ExecutionPolicy Bypass -File "%~dp0studio.ps1" %*
